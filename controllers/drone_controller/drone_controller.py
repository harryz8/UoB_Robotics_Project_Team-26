from mapping import Mapping
"""drone_controller controller."""

# You may need to import some classes of the controller module. Ex:
#  from controller import Robot, Motor, DistanceSensor
from controller import Robot
from controller import Keyboard

BLOCK_LENGTH = 350 #mm

# create the Robot instance.
robot = Robot()

# get the time step of the current world.
timestep = int(robot.getBasicTimeStep())

# create keyboard instance
keyboard=Keyboard()
keyboard.enable(timestep)

# create the map
mapping = Mapping(BLOCK_LENGTH)

# You should insert a getDevice-like function in order to get the
# instance of a device of the robot. Something like:
#  motor = robot.getDevice('motorname')
#  ds = robot.getDevice('dsname')
#  ds.enable(timestep)

motors = []
motors.append(robot.getDevice("front left propeller"))
motors.append(robot.getDevice("front right propeller"))
motors.append(robot.getDevice("rear left propeller"))
motors.append(robot.getDevice("rear right propeller"))

camera = robot.getDevice("camera")
camera.enable(timestep)

lidar = robot.getDevice("lidar")
lidar.enable(timestep)
lidar.enablePointCloud()

# Main loop:
# - perform simulation steps until Webots is stopping the controller
while robot.step(timestep) != -1:
    key=keyboard.getKey()
    if (key == "w"):
        pass
        #go forward
    #...
    elif (key == "r"):
        pass
        # Ben
        
    #localisation -> mapping -> databse of map
    pass

# Enter here exit cleanup code.
