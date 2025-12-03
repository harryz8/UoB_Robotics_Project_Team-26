"""drone_controller controller."""
#from mapping import Mapping
#from path_planner import Path_Planner
from Ava_Drone_localization import *

from controller import Robot, Motor, GPS, InertialUnit, Gyro
from controller import Keyboard
import math
import numpy as np

BLOCK_LENGTH : int = 350 #mm

#clamps values
def clamp(value, low, high):
    return max(low, min(value, high))

# create the Robot instance.
robot = Robot()

# get the time step of the current world.
timestep = int(robot.getBasicTimeStep())

# create keyboard instance
keyboard=Keyboard()
keyboard.enable(timestep)

# create the map
#mapping : Mapping = Mapping(BLOCK_LENGTH)
#path_planner : Path_Planner = Path_Planner()

# You should insert a getDevice-like function in order to get the
# instance of a device of the robot. Something like:
#  motor = robot.getDevice('motorname')
#  ds = robot.getDevice('dsname')
#  ds.enable(timestep)

# getting all motors
front_left_motor = robot.getDevice("front left propeller")
front_right_motor = robot.getDevice("front right propeller")
rear_left_motor = robot.getDevice("rear left propeller")
rear_right_motor = robot.getDevice("rear right propeller")

motors = [front_left_motor, front_right_motor, rear_left_motor, rear_right_motor]

#Set motors to velocity control
for motor in motors:
    motor.setPosition(float('inf')) # set to infinity = velocity control
    motor.setVelocity(0.0)
    
# PID constants for hover
#controller gain for each axis, i.e. the higher vertical_offset the more aggressively it tries to reach target altitiude
vertical_thrust_base = 68.5       
vertical_offset = 0.6            
vertical_gain = 3.0              
roll_gain = 50.0                  
pitch_gain = 30.0
yaw_gain = 15.0

target_altitude = 1.0
target_yaw = 0.0
    
# getting camera device
camera = robot.getDevice("camera")
camera.enable(timestep)

# getting lidar device
lidar = robot.getDevice("lidar")
lidar.enable(timestep)
lidar.enablePointCloud()

#get inertial unit
imu = robot.getDevice("inertial unit")
imu.enable(timestep)

#get gps
gps = robot.getDevice("gps")
gps.enable(timestep)

#get gyro
gyro = robot.getDevice("gyro")
gyro.enable(timestep)

#----localization (particle filtering) initialization------
N=500 #number of particles
space=(-5,5,-5,5,0,10) #x_min,x_max,y_min,y_max,z_min,z_max
particles_position,weight=initial_particles(N,space)
#store previous GPS position for velocity computation
prev_gps=gps.getValues()
timestep_=timestep/1000.0 # convert ms to seconds



# Main loop:
# - perform simulation steps until Webots is stopping the controller
while robot.step(timestep) != -1:
    key=keyboard.getKey()
    
    # read sensors
    roll, pitch, yaw = imu.getRollPitchYaw()
    gps_values=gps.getValues()
    altitude = gps.getValues()[2]
    roll_velocity, pitch_velocity, yaw_velocity= gyro.getValues()
    #-----localization:particle filter update------
    #compute drone velocity in world frame from GPS difference
    curr_gps=np.array(gps_values)
    prev_gps_np=np.array(prev_gps)
    v_world=(curr_gps-prev_gps_np)/timestep_
    prev_gps=gps_values #update for next step
    #convert velocity to body frame using current orientation
    R=orientation_angle_matrix(yaw,pitch,roll)
    drone_velocity=R.T@v_world #body frame velocity
    # angular velocity from gyro
    ang_velocity=np.array([roll_velocity,pitch_velocity,yaw_velocity])
    #prediction step
    prediction_step(particles_position,weight,drone_velocity,ang_velocity,timestep_)
    #sensor updates
    gps_update(particles_position,weight,gps_values)
    compass_update(particles_position,weight,yaw)
    #resampling
    Neff=important_particles(weight)
    if Neff<N/2:
        resample(particles_position,weight)
    #Final state estimation
    pf_position,pf_orientation=final_estimation(particles_position,weight)   
    print("GPS:", np.round(gps_values,3),"PF Pose:",np.round(pf_position,3),"yaw:",round(yaw,3),"PF yaw:",round(float(pf_orientation[0]),3))   
    
    
    
    # stabilization
    roll_input = roll_gain * clamp(roll, -1.0, 1.0)
    pitch_input = pitch_gain * clamp(pitch, -1.0, 1.0)
    vertical_input = vertical_gain * (clamp(target_altitude - altitude + vertical_offset, -1.0, 1.0)) ** 3
    
     # wrap yaw error to [-pi, pi]
    yaw_error = math.atan2(math.sin(target_yaw - yaw), math.cos(target_yaw - yaw))
    yaw_input = yaw_gain * clamp(yaw_error, -1.0, 1.0)

    # Motor velocities including yaw
    fl_input = vertical_thrust_base + vertical_input - roll_input + pitch_input - yaw_input
    fr_input = vertical_thrust_base + vertical_input + roll_input + pitch_input + yaw_input
    rl_input = vertical_thrust_base + vertical_input - roll_input - pitch_input + yaw_input
    rr_input = vertical_thrust_base + vertical_input + roll_input - pitch_input - yaw_input
    if (key == ord("W")):
        pass
    
    #...
    elif (key == ord("R")):
       # print(path_planner.test())
        pass
        # Ben
        
    #localisation -> mapping -> databse of map
    pass
    
     # Apply velocities
    front_left_motor.setVelocity(fl_input)
    front_right_motor.setVelocity(-fr_input)
    rear_left_motor.setVelocity(-rl_input)
    rear_right_motor.setVelocity(rr_input)

# Enter here exit cleanup code.
