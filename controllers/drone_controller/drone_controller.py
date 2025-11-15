"""drone_controller controller."""
from mapping import Mapping
from path_planner import Path_Planner
import numpy as np
import time
# You may need to import some classes of the controller module. Ex:
#  from controller import Robot, Motor, DistanceSensor
from controller import Robot
from controller import Keyboard

BLOCK_LENGTH : int = 350 #mm

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
keyboard=Keyboard()
keyboard.enable(timestep)

# create the map
mapping : Mapping = Mapping(BLOCK_LENGTH)
path_planner : Path_Planner = Path_Planner()

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

# Main loop:
# - perform simulation steps until Webots is stopping the controller
while robot.step(timestep) != -1:
    key=keyboard.getKey()
    if (key == ord("W")):
        pass
        #go forward
    #...
    elif (key == ord("R")):
        print(path_planner.get_Path(path_planner.shortest_path((0,0,0),(2,4,4),np.array(grid)),(2,4,4)))
        time.sleep(1)
        pass
        # Ben
        
    #localisation -> mapping -> databse of map
    pass

# Enter here exit cleanup code.
