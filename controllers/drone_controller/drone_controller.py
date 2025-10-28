from mapping import Mapping
"""drone_controller controller."""

# You may need to import some classes of the controller module. Ex:
#  from controller import Robot, Motor, DistanceSensor
from controller import Robot
from controller import Keyboard

# create the Robot instance.
robot = Robot()

# get the time step of the current world.
timestep = int(robot.getBasicTimeStep())

# create keyboard instance
keyboard=Keyboard()
keyboard.enable(timestep)

# create the map
mapping = Mapping()

# You should insert a getDevice-like function in order to get the
# instance of a device of the robot. Something like:
#  motor = robot.getDevice('motorname')
#  ds = robot.getDevice('dsname')
#  ds.enable(timestep)

# Main loop:
# - perform simulation steps until Webots is stopping the controller
while robot.step(timestep) != -1:
    key=keyboard.getKey()
    if (key == "w"):
        pass
        #go forward
    #...
    elif (key == "r"):
        # Ben
        
    #localisation -> mapping -> databse of map
    pass

# Enter here exit cleanup code.
