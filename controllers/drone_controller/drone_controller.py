"""drone_controller controller."""
from mapping import *
from path_planner import Path_Planner
from lidar import Lidar
import numpy as np
import math

from controller import Robot, Motor, GPS, InertialUnit, Gyro
from controller import Keyboard

BLOCK_LENGTH: float = 350  # The length of one side of the cube shaped 'block' which the world is split into in the map. In mm
ROBOT_SIZE: int = 1  # How many blocks the lidar takes up. In blocks

#clamps values
def clamp(value, low, high):
    return max(low, min(value, high))

# create the Robot instance.
robot = Robot()

# get the time step of the current world.
timestep = int(robot.getBasicTimeStep())

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
mapping_inst: Mapping = Mapping(BLOCK_LENGTH, ROBOT_SIZE, (100, 100, 100))

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
    mapping_inst.update(np.array(gps.getValues()), np.array(gyro.getValues()), horizontal_lidar)
    mapping_inst.update(np.array(gps.getValues()), np.array(gyro.getValues()), vertical_lidar)
    
    # read sensors
    roll, pitch, yaw = imu.getRollPitchYaw()
    altitude = gps.getValues()[2]
    roll_velocity, pitch_velocity, _ = gyro.getValues()

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
    elif key == ord("R"):
        # something like this
        """ path_planner.shortest_path(meters_to_blocks(localisation.get_coordinates(), BLOCK_LENGTH),
                                   mapping_inst.origin,
                                   mapping_inst.get()) """
        print(path_planner.test())
        pass
        # Ben
    # For debugging
    if (prints > 0) and (loops % prints == 0):
        pass
        # print(f"{mapping_inst.get_normalised(maximum_certainty_log_odds=10000)}\n\r\n\r")  # maximum_certainty_log_odds to be determined
        # print(mapping_inst.get().shape)
    loops += 1

    #localisation -> mapping -> database of map
    pass
    
     # Apply velocities
    front_left_motor.setVelocity(fl_input)
    front_right_motor.setVelocity(-fr_input)
    rear_left_motor.setVelocity(-rl_input)
    rear_right_motor.setVelocity(rr_input)

# Enter here exit cleanup code.
