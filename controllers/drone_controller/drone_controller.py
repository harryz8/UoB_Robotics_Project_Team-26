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
x_gain = 0.75

target_altitude = 1.0 #altitude it tries to reach
target_yaw = 0.0 #yaw it tries to reach

#x-axis PID, forward/back
target_x = 0.0
x_kp = 0.8 #x proportional gain
x_ki = 0.0 #x integral gain
x_kd = 0.2 #x derivative gain
x_integral = 0.0 #store state for calc
x_prev_error = 0.0 #store state for calc
x_velocity_gain = 0.5 #adding in damping

#y-axis PID, right/left, same as above but with y respectively
target_y = 0.0
y_kp = 0.8
y_ki = 0.0
y_kd = 0.2
y_integral = 0.0
y_prev_error = 0.0
y_velocity_gain = 0.5

#maximum allowed values for pitch or roll correction
max_pitch_correction = 0.5 
max_roll_correction = 0.5
    
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

#prev gps values for velocity stabilisation
x_prev = gps.getValues()[0]
y_prev = gps.getValues()[1]

# Main loop:
# - perform simulation steps until Webots is stopping the controller
while robot.step(timestep) != -1:
    key=keyboard.getKey()
    
    # Read sensors
    roll, pitch, yaw = imu.getRollPitchYaw()
    x_current = gps.getValues()[0]
    y_current = gps.getValues()[1]
    altitude = gps.getValues()[2]

    # Calculate x and y velocities
    dt = timestep / 1000.0
    x_velocity = (x_current - x_prev) / dt
    y_velocity = (y_current - y_prev) / dt
    x_prev = x_current
    y_prev = y_current

    # Vertical stabilization (altitude)
    vertical_input = vertical_gain * (clamp(target_altitude - altitude + vertical_offset, -1.0, 1.0)) ** 3

    # X-axis PID for pitch
    x_error = target_x - x_current
    x_integral += x_error * dt
    x_derivative = (x_error - x_prev_error) / dt
    x_prev_error = x_error

    pitch_correction = x_kp * x_error + x_ki * x_integral + x_kd * x_derivative
    pitch_correction -= x_velocity_gain * x_velocity
    pitch_correction = clamp(pitch_correction, -max_pitch_correction, max_pitch_correction)

    # Y-axis PID for roll
    y_error = target_y - y_current
    y_integral += y_error * dt
    y_derivative = (y_error - y_prev_error) / dt
    y_prev_error = y_error

    roll_correction = y_kp * y_error + y_ki * y_integral + y_kd * y_derivative
    roll_correction -= y_velocity_gain * y_velocity
    roll_correction = clamp(roll_correction, -max_roll_correction, max_roll_correction)

    # combines IMU with PID for stabilisation
    roll_input = roll_gain * clamp(roll, -1.0, 1.0) + roll_correction
    pitch_input = pitch_gain * clamp(pitch, -1.0, 1.0) + pitch_correction

    # pitch correction, found through iterative testing, multiplies roll by x and adds it to pitch
    pitch_input += -7.746551065262 * roll 

    # Yaw stabilization
    yaw_error = math.atan2(math.sin(target_yaw - yaw), math.cos(target_yaw - yaw))
    yaw_input = yaw_gain * clamp(yaw_error, -1.0, 1.0)

    # Motor mixing, essentially computes each motors thrust based on base, vertical, roll, pitch and yaw
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
