"""Ava_Drone_localization controller."""

import numpy as np

def angle_correction(angle):
    return (a+np.pi)%(2*np.pi)-np.pi

def orientation_angle_matrix(yaw,pitch,roll):
    cos_y=np.cos(yaw)
    sin_y=np.sin(yaw)
    cos_p=np.cos(pith)
    sin_p=np.sin(pith)
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
    return Rotation_yaw@Rotation_pitch@Rotation_roll

def weight_normalize(weight):
    sum=weight.sum()
    if sum<=1e-12:
        weight[:]=1.0/len(weight)
    else:
        weight[:]=weight/sum

def correct_mean(angle,weight):
    sin_sum=np.sum(np.sin(angle)*weight)
    cos_sum=np.sum(np.cos(angle)*weight)
    return np.arctan2(s,c)
     
def initial_particles(N,space):
    random=np.random.default_random()
    x_min,x_max,y_min,y_max,z_min,z_max=space
    particles_position=np.zeros((N,6),float)
    particles_position[:,0]=random.uniform(x_min,x_max,N)
    particles_position[:,1]=random.uniform(y_min,y_max,N)
    particles_position[:,2]=random.uniform(z_min,z_max,N)
    particles_position[:,3]=random.uniform(-np.pi,np.pi,N)
    particles_position[:,4]=0.0
    particles_position[:,5]=0.0
    weight=np.ones()/N
    return particles_position,weight
    
def prediction_step(particles_position,weight,drone_velocity,ang_velocity,timestep):
    particles_position[:,3]=angle_correction(particles_position[:,3]+ang_velocity[2]*timestep)
    particles_position[:,4]=angle_correction(particles_position[:,4]+ang_velocity[1]*timestep)
    particles_position[:,5]=angle_correction(particles_array[:,5]+ang_velocity[0]*timestep)
    #move particles!
    for i in range(N):
        R=orientation_angle_matrix(particles_position[i,3],particles_position[i,4],particles_position[i,5])
        velocity_in_world=R@np.array(drone_velocity,float)
    
        particles_angle[i,0:3]= particles_angle[i,0:3]+(velocity_in_world*timestep)
#now we update sensors!

def gps_update(particles_position,weight,gps_data,std=(2.0,2.0,5.0)):
    gps_x,gps_y,gps_z=map(float,gps_data)
    stan_devi_x,stan_devi_y,stan_devi_z=map(float,std)
    variance_x=stan_devi_x**2
    variance_y=stan_devi_y**2
    variance_z=stan_devi_z**2 
    for i in range(particles_position.shape[0]):
        x_difference= abs(gps_x - particles_position[i,0])
        y_difference= abs(gps_y - particles_position[i,1])
        z_difference= abs(gps_z - particles_position[i,2])
        power=-0.5*(x_difference**2/variance_x + y_difference**2/variance_y+ z_difference**2/variance_z)
        weight[i]=weight[i]*exp(power)   
    weight_normalize(weight)  
       
         
def compass_update( particles_position,weight,compass_data,yaw_std=0.2):
    yaw_variance=yaw_std**2
    for i in range(particles_position.shape[0]):
        yaw_differenc=angle_correction(compass_data-particles_position[i,3]) 
        weight[i]=weight[i]*np.exp(-0.5*(yaw_differenc**2)/yaw_variance) 
    weight_normalize(weight) 
       
def important_particles(weight):
    sum_of_sw=np.sum(weight**2)
    if sum_of_sw==0:
        return 0
    return (1.0/sum_of_sw)
    
    
def resample(particles_position,weight):
    N=lenght(weight)
    weight=weight/np.sum(weight)
    weight_cumulative_sum=np.cumsum(weight)
    best_particles=np.zeros_like(particles_position)
    for i in range(N):
        random_number=np.random.rand()
        bin=np.searchsorted(weight_cumulative_sum,random_number)
        best_particles[i]=particles_position[bin]
    particles_position[:]=best_particles
    weight[:]=1.0/N    
    
def final_estimation(particles_position,weight):
    weight=weight/(weight.sum()
    
    x=np.sum(particles_position[:,0] * weight)
    y=np.sum(particles_position[:,1] * weight)
    z=np.sum(particles_position[:,2] * weight)
    position=np.array([x,y,z])
    yaw=correct_mean(particles_position[:,3],weight)
    pitch=correct_mean(particles_position[:,4],weight)
    roll=correct_mean(particles_position[:,5],weight)
    orientation=np.array([yaw,pitch,roll])
    return position,orentation

def hello():
    print("Hello")






















                          
                   
                         
    