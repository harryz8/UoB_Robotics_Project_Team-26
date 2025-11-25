"""drone_controller controller."""
from mapping import Mapping, meters_to_blocks
from path_planner import Path_Planner
# You may need to import some classes of the controller module. Ex:
#  from controller import Robot, Motor, DistanceSensor
from controller import Robot, Motor, GPS, InertialUnit, Gyro, Keyboard
from lidar import Lidar
import numpy as np
import math
import threading

BLOCK_LENGTH: float = 350  # The length of one side of the cube shaped 'block' which the world is split into in the map. In mm
ROBOT_SIZE: int = 1  # How many blocks the lidar takes up. In blocks

#clamps values
def clamp(value, low, high):
    return max(low, min(value, high))

# create the Robot instance.
robot = Robot()

# get the time step of the current world.
timestep = int(robot.getBasicTimeStep())
grid = [
    [   
        [1,  1,  1,  0, -1],
        [1,  1,  0,  0, -1],
        [1,  1,  1,  1, -1],
        [0,  0,  1,  1,  1],
        [-1, -1, 1,  1,  1]
    ],
    [   
        [1,  1,  1,  1,  1],
        [1,  0,  0,  1, -1],
        [1,  1,  1,  1, -1],
        [1,  0,  1,  1,  1],
        [-1, -1, 1,  0,  -1]
    ],
    [   
        [1,  1,  0,  0,  0],
        [1,  1,  1,  1, -1],
        [0,  1,  1,  1, -1],
        [0,  0,  1,  1,  1],
        [-1,  1,  1,  1,  1]
    ]
]
# create keyboard instance
keyboard = Keyboard()
keyboard.enable(timestep)

path_planner: Path_Planner = Path_Planner()

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

#Set motors to velocity control, makes it so motors dont try to reach specific angle
for motor in motors:
    motor.setPosition(float('inf')) # set to infinity = velocity control
    motor.setVelocity(0.0)
    
# constants for hover
# gain is multiplied by error for stabilization
vertical_thrust_base = 68.5    #base thrust given to each motor  
vertical_offset = 0.6       #bias added to altitude correction   
vertical_gain = 3.0              #gain for altitude
roll_gain = 50.0            #gain for roll      
pitch_gain = 30.0           #etc...
yaw_gain = 15.0
#x_gain = 0.75

target_altitude = 1.0 #altitude it tries to reach
target_yaw = 0.0 #yaw it tries to reach

#x-axis PID, forward/back, no longer used but keep just in case
#target_x = 0.0
#x_kp = 0.8 #x proportional gain
#x_ki = 0.0 #x integral gain
#x_kd = 0.2 #x derivative gain
#x_integral = 0.0 #store state for calc
#x_prev_error = 0.0 #store state for calc
#x_velocity_gain = 0.5 #adding in damping

#y-axis PID, right/left, same as above but with y respectively
#target_y = 0.0
#y_kp = 0.8
#y_ki = 0.0
#y_kd = 0.2
#y_integral = 0.0
#y_prev_error = 0.0
#y_velocity_gain = 0.5

#keyboard movement stuff
key_increment = 0.03  # movement per step
key_damping = 0.95    # damping when no key pressed
x_offset = 0.0
y_offset = 0.0
yaw_offset = 0.0
yaw_increment = 0.02
yaw_damping = 0.90
altitude_increment = 0.01
altitude_max = 10.0
altitude_min = 0.2 #stop it crashing into ground

#maximum allowed values for pitch or roll correction
max_pitch_correction = 0.8 
max_roll_correction = 0.8

#drift correction, values found through iterative testing
x_trim = 0.15
y_trim = 0.064
    
# getting camera device
camera = robot.getDevice("camera")
camera.enable(timestep)

# getting lidar devices and initialising lidar objects
horizontal_lidar_device = robot.getDevice("horizontal_lidar")
horizontal_lidar_device.enable(timestep)
horizontal_lidar: Lidar = Lidar(horizontal_lidar_device,
                          axis_from_robot=(0, 1),
                          object_detected_given_object_prob=0.9,  # to be determined
                          empty_detected_given_empty_prob=0.9  # to be determined
                          )
vertical_lidar_device = robot.getDevice("vertical_lidar")
vertical_lidar_device.enable(timestep)
vertical_lidar: Lidar = Lidar(vertical_lidar_device,
                          axis_from_robot=(0, 2),
                          object_detected_given_object_prob=0.9,  # to be determined
                          empty_detected_given_empty_prob=0.9  # to be determined
                          )

# create the map in a Mapping object
mapping_inst: Mapping = Mapping(BLOCK_LENGTH, ROBOT_SIZE)

#get inertial unit
imu = robot.getDevice("inertial unit")
imu.enable(timestep)

#get gps
gps = robot.getDevice("gps")
gps.enable(timestep)

#get gyro
gyro = robot.getDevice("gyro")
gyro.enable(timestep)

# for debugging
loops = 0
prints = 10
np.set_printoptions(edgeitems=30, linewidth=100000,
                    formatter=dict(float=lambda x: "%.3g" % x))

# Main loop:
# - perform simulation steps until Webots is stopping the controller
while robot.step(timestep) != -1:
    key = keyboard.getKey()

    # Update the map given readings from both LIDARs
    # Using threads to speed up the process by running many operations in parallel
    step_update_threads = []
    step_update_threads.append(threading.Thread(target=mapping_inst.update,
                                               args=(np.array(gps.getValues()), np.array(gyro.getValues()),
                                                     horizontal_lidar)))
    step_update_threads.append(threading.Thread(target=mapping_inst.update,
                                               args=(np.array(gps.getValues()), np.array(gyro.getValues()),
                                                     vertical_lidar)))
    for update_thread in step_update_threads:
        update_thread.start()
    # wait for threads to finish
    for update_thread in step_update_threads:
        update_thread.join()

    # read sensors

    roll, pitch, yaw = imu.getRollPitchYaw()
    altitude = gps.getValues()[2]
    
    # Keyboard input
    pressed_keys = set()
    k = keyboard.getKey()
    while k != -1:
        pressed_keys.add(k)
        k = keyboard.getKey()

    # Keyboard offsets
    if ord("W") in pressed_keys:
        x_offset += key_increment   # forward
    if ord("S") in pressed_keys:
        x_offset -= key_increment   # backward
    if ord("A") in pressed_keys:
        y_offset -= key_increment   # left
    if ord("D") in pressed_keys:
        y_offset += key_increment   # right
    if Keyboard.LEFT in pressed_keys:
        yaw_offset += yaw_increment   #rotate left
    if Keyboard.RIGHT in pressed_keys:
        yaw_offset -= yaw_increment    #rotate right
    if Keyboard.UP in pressed_keys:
        target_altitude += altitude_increment
    if Keyboard.DOWN in pressed_keys:
        target_altitude -= altitude_increment
        
    # Clamp altitude to safe range
    target_altitude = clamp(target_altitude, altitude_min, altitude_max)

    # Apply damping
    x_offset *= key_damping
    y_offset *= key_damping
    yaw_offset *= yaw_damping
    
    #updates target yaw to allow for user input
    target_yaw += yaw_offset * 0.05

    # Vertical stabilization (altitude)
    vertical_input = vertical_gain * (clamp(target_altitude - altitude + vertical_offset, -1.0, 1.0)) ** 3

    # calculate pitch/roll correction with keyboard ofssets 
    pitch_correction = clamp(x_offset * 8.0 + x_trim, -max_pitch_correction, max_pitch_correction)
    roll_correction  = clamp(y_offset * 8.0 + y_trim, -max_roll_correction, max_roll_correction)
    
    # Combine with IMU stabilization
    roll_input  = roll_gain * clamp(roll, -1.0, 1.0) + roll_correction
    pitch_input = pitch_gain * clamp(pitch, -1.0, 1.0) + pitch_correction
    #pitch_input += -7.746551065262 * roll  # iterative pitch fix

    # Yaw stabilization
    yaw_error = math.atan2(math.sin(target_yaw - yaw), math.cos(target_yaw - yaw))
    yaw_input = yaw_gain * clamp(yaw_error, -1.0, 1.0)

    # Motor mixing, essentially computes each motors thrust based on base, vertical, roll, pitch and yaw
    fl_input = vertical_thrust_base + vertical_input - roll_input + pitch_input - yaw_input
    fr_input = vertical_thrust_base + vertical_input + roll_input + pitch_input + yaw_input
    rl_input = vertical_thrust_base + vertical_input - roll_input - pitch_input + yaw_input
    rr_input = vertical_thrust_base + vertical_input + roll_input - pitch_input - yaw_input

     # Apply velocities
    front_left_motor.setVelocity(fl_input)
    front_right_motor.setVelocity(-fr_input)
    rear_left_motor.setVelocity(-rl_input)
    rear_right_motor.setVelocity(rr_input)

    if ord("R") in pressed_keys:#start end map, end
        print(path_planner.get_Path(path_planner.shortest_path(meters_to_blocks(gps.getValues(),BLOCK_LENGTH), mapping_inst.origin, mapping_inst.get_normalised(1000)), mapping_inst.origin))

# Enter here exit cleanup code.
