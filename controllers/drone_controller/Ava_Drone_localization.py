"""Ava_Drone_localization controller."""

import numpy as np

def angle_correction(angle): #wrap angle to [-pi,+pi]
    return (angle+np.pi)%(2*np.pi)-np.pi

def orientation_angle_matrix(yaw,pitch,roll): #Rotation Matrix
# builds a 3*3 rotation matrix from yaw(z-axis),pitch(y-axis),roll(x-axis)
    cos_y=np.cos(yaw)
    sin_y=np.sin(yaw)
    cos_p=np.cos(pitch)
    sin_p=np.sin(pitch)
    cos_r=np.cos(roll)
    sin_r=np.sin(roll)
    Rotation_roll= np.array([[1,  0   ,   0   ],
                             [0,  cos_r,-sin_r],
                             [0,  sin_r,cos_r ]])
    Rotation_pitch= np.array([[cos_p , 0, sin_p],
                             [0,      1,   0   ],
                             [-sin_p, 0, cos_p]])
    Rotation_yaw= np.array([[cos_y ,-sin_y, 0 ],
                            [sin_y , cos_y ,0 ], 
                            [0,       0,    1 ]])
    return Rotation_yaw@Rotation_pitch@Rotation_roll #mutliply three matrix
    #resultl: a matrix that transforms vectors from drone's body frame to worl coordinate

#----- Weight Normalization___
def weight_normalize(weight): #normalize particle weights: sum=1
    sum_weight=weight.sum() 
    if sum_weight<=1e-12 or not np.isfinite(sum_weight):#sum is almost zero, dividing impossible!
        weight[:]=1.0/len(weight) #reset all weights to be uniform(all equal)
    else:
        weight[:]=weight/sum_weight #devide each weight by the total sum: sum(weight)==1

#---------Circular mean of angles--------
def correct_mean(angle,weight): #compute weighted mean of angles
    sin_sum=np.sum(np.sin(angle)*weight)
    cos_sum=np.sum(np.cos(angle)*weight)
    return np.arctan2(sin_sum,cos_sum) #mean anglenin correct quadrant
    
#--------------Particle Initialization----------- 
def initial_particles(N,space): #initialize N particles in space 
    rng=np.random.default_rng()
    x_min,x_max,y_min,y_max,z_min,z_max=space #space: list (6) defininf box limit
    particles_position=np.zeros((N,6),float) # array an N*6 array
    #Position
    particles_position[:,0]=rng.uniform(x_min,x_max,N) #random x position in x_min , x_max
    particles_position[:,1]=rng.uniform(y_min,y_max,N)#random y position in y_min , y_max
    particles_position[:,2]=rng.uniform(z_min,z_max,N)#random z position in z_min , z_max
    #orientation
    particles_position[:,3]=rng.uniform(-np.pi,np.pi,N) #random yaw angle in [-pi,+pi]
    particles_position[:,4]=0.0 #all particles start with pitch=0
    particles_position[:,5]=0.0#all particles start with roll=0
    weight=np.ones(N,dtype=float)/N #weight array
    return particles_position,weight 
 
#------------------Prediction Step-----------    
def prediction_step(particles_position,weight,drone_velocity,ang_velocity,timestep):
    #move particles with dron velocity,update orientation
    particles_position[:,3]=angle_correction(particles_position[:,3]+ang_velocity[2]*timestep)#new yaw: old yaw+yaw_rate*timestep, wrapped [-pi,+pi]
    particles_position[:,4]=angle_correction(particles_position[:,4]+ang_velocity[1]*timestep)#pitch same as above
    particles_position[:,5]=angle_correction(particles_position[:,5]+ang_velocity[0]*timestep)#roll same as above
    N=particles_position.shape[0]
    velocity_body=np.array(drone_velocity,float)
    if not np.all(np.isfinite(velocity_body)):
        return
    #move particles!
    for i in range(N):#loop over all particles
        yaw,pitch,roll=particles_position[i,3:6]#extraxt orientation of particle i
        R=orientation_angle_matrix(yaw,pitch,roll) #build rotation matrix from body frame to world frame
        velocity_in_world=R@velocity_body #convert velocity from drone frame to world frame
        particles_position[i,0:3]= particles_position[i,0:3]+(velocity_in_world*timestep) #update position: x_new=x_old+velocity world*timestep
    particles_position[:]=np.nan_to_num(particles_position,nan=0.0,posinf=0.0,neginf=0.0)

#-----------now we update sensors!-------
#---------GPS Update
def gps_update(particles_position,weight,gps_data,std=(0.3,0.3,0.6)): #update particle weights based on GPS measurments, std: standard deviation of GPS noise
    gps_x,gps_y,gps_z=map(float,gps_data) #read GPS measurments
    stan_devi_x,stan_devi_y,stan_devi_z=std #sigmax,sigmay,sigmaz
    variance_x=stan_devi_x**2 #variance=sigma^2
    variance_y=stan_devi_y**2
    variance_z=stan_devi_z**2 
    for i in range(particles_position.shape[0]):#loop over particles
        x_difference= gps_x - particles_position[i,0] #difference in x between GPS and particle position
        y_difference= gps_y - particles_position[i,1]#difference in y
        z_difference= gps_z - particles_position[i,2]#difference in z
        power=-0.5*(x_difference**2/variance_x + y_difference**2/variance_y+ z_difference**2/variance_z) #exponent part of Guassian
        weight[i]=weight[i]*np.exp(power)  #multiply particle weight by likelihood  
    weight_normalize(weight)  
       
#----------Compass Update         
def compass_update(particles_position,weight,compass_data,yaw_std=0.2):
    yaw_variance=yaw_std**2# variance of yaw noise
    for i in range(particles_position.shape[0]):
        yaw_differenc=angle_correction(compass_data-particles_position[i,3]) #difference between measured yaw and particle yaw
        weight[i]=weight[i]*np.exp(-0.5*(yaw_differenc**2)/yaw_variance) #multiply weight by Guassian likelihood based on yaw error
    weight_normalize(weight) 
    
#----eefective particle count------     
def important_particles(weight):#compute thr effective number of particles, how mnany particles are effectively used
    sum_of_sw=np.sum(weight**2)
    if sum_of_sw<=0 or not np.isfinite(sum_of_sw):#avoide deviding by zero
        return 0.0
    return (1.0/sum_of_sw)
    
#----resampling!----    
def resample(particles_position,weight):#resample particles according to their weights
    N=len(weight)
    weight_normalize(weight)
    cdf=np.cumsum(weight)
    new_particles=np.zeros_like(particles_position)
    for i in range(N):#for each new particle
        random_number=np.random.rand() #draw a random number
        bin=np.searchsorted(cdf,random_number)# find which particle this random number selects based on CDF
        new_particles[i]=particles_position[bin]#compy that particle 
    particles_position[:]=new_particles #replace old particles with resamples ones
    weight[:]=1.0/N   
 
#----------------Final state estimation -------------  
def final_estimation(particles_position,weight): #compute final estimated position and orientation
    particles_clean=np.nan_to_num(particles_position,nan=0.0,posinf=0.0,neginf=0.0)
    weight_p=weight/(weight.sum()+1e-300) #avoid NAN
    
    x=np.sum(particles_position[:,0] * weight_p) #weighted average of x position
    y=np.sum(particles_position[:,1] * weight_p)#weighted average of y position
    z=np.sum(particles_position[:,2] * weight_p)#weighted average of z position
    position=np.array([x,y,z])
    
    yaw=correct_mean(particles_position[:,3],weight_p)#weighted cirvular mean of yaw
    pitch=correct_mean(particles_position[:,4],weight_p)#weighted cirvular mean of pitch
    roll=correct_mean(particles_position[:,5],weight_p)#weighted cirvular mean of roll
    orientation=np.array([yaw,pitch,roll]) #final orientation estimate [yaw,pitch,roll]
    
    return position,orientation #output : (position, orientation)























                          
                   
                         
    