"""drone_controller controller."""
from mapping import Mapping

# You may need to import some classes of the controller module. Ex:
#  from controller import Robot, Motor, DistanceSensor
from controller import Robot
from controller import Keyboard

BLOCK_LENGTH : float = 350 #mm
ROBOT_SIZE : int = 1 #block

# create the Robot instance.
robot = Robot()

# get the time step of the current world.
timestep = int(robot.getBasicTimeStep())

# create keyboard instance
keyboard=Keyboard()
keyboard.enable(timestep)

# You should insert a getDevice-like function in order to get the
# instance of a device of the robot. Something like:
#  motor = robot.getDevice('motorname')
#  ds = robot.getDevice('dsname')
#  ds.enable(timestep)

# getting all motors
motors = []
motors.append(robot.getDevice("front left propeller"))
motors.append(robot.getDevice("front right propeller"))
motors.append(robot.getDevice("rear left propeller"))
motors.append(robot.getDevice("rear right propeller"))

# getting camera device
camera = robot.getDevice("camera")
camera.enable(timestep)

# getting lidar device
lidar = robot.getDevice("lidar")
lidar.enable(timestep)
lidar.enablePointCloud()

# create the map
mapping : Mapping = Mapping(BLOCK_LENGTH, ROBOT_SIZE, lidar)

# Main loop:
# - perform simulation steps until Webots is stopping the controller
while robot.step(timestep) != -1:
    key=keyboard.getKey()
    mapping.update()

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
