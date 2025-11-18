"""drone_controller controller."""
#from mapping import Mapping
#from path_planner import Path_Planner

from controller import Robot, Motor, GPS, InertialUnit, Gyro
from controller import Keyboard
import math

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
x_gain = 0.75

target_altitude = 1.0
target_yaw = 0.0

target_x = 0.0
x_kp = 0.3
x_ki = 999.9
x_kd = 0.05

x_integral = 0.0
x_prev_error = 0.0
max_pitch_correction = 0.1
    
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

#prev x coordinate for velocity calc
x_prev = gps.getValues()[0]

# Main loop:
# - perform simulation steps until Webots is stopping the controller
while robot.step(timestep) != -1:
    key=keyboard.getKey()
    
    # read sensors
    roll, pitch, yaw = imu.getRollPitchYaw()
    altitude = gps.getValues()[2]

    # stabilization
    roll_input = roll_gain * clamp(roll, -1.0, 1.0)
    pitch_input = pitch_gain * clamp(pitch, -1.0, 1.0)
    vertical_input = vertical_gain * (clamp(target_altitude - altitude + vertical_offset, -1.0, 1.0)) ** 3

    # --- X-axis PID control ---
    x_current = gps.getValues()[0]
    x_error = target_x - x_current

    x_integral += x_error
    x_derivative = x_error - x_prev_error
    x_prev_error = x_error

    # PID output in radians for safe pitch correction
    pitch_correction = x_kp * x_error + x_ki * x_integral + x_kd * x_derivative
    pitch_correction = clamp(pitch_correction, -max_pitch_correction, max_pitch_correction)

    pitch_input += pitch_correction

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
        print(path_planner.test())
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
