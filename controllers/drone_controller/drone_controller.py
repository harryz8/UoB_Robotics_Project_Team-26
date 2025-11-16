"""drone_controller controller."""
from mapping import *
from path_planner import Path_Planner
from lidar import Lidar
import numpy as np

# You may need to import some classes of the controller module. Ex:
#  from controller import Robot, Motor, DistanceSensor
from controller import Robot
from controller import Keyboard

BLOCK_LENGTH: float = 350  # mm
ROBOT_SIZE: int = 1  # block

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
motors = [robot.getDevice("front left propeller"), robot.getDevice("front right propeller"),
          robot.getDevice("rear left propeller"), robot.getDevice("rear right propeller")]

# getting camera device
camera = robot.getDevice("camera")
camera.enable(timestep)

# getting lidar device
lidar_device = robot.getDevice("lidar")
lidar_device.enable(timestep)
lidar_inst: Lidar = Lidar(lidar_device,
                          axis_from_robot=(0, 1),
                          object_detected_given_object_prob=0.9,  # to be determined
                          empty_detected_given_empty_prob=0.9  # to be determined
                          )

# create the map
mapping_inst: Mapping = Mapping(BLOCK_LENGTH, ROBOT_SIZE)

# for debugging
loops = 0
prints = 10

# Main loop:
# - perform simulation steps until Webots is stopping the controller
while robot.step(timestep) != -1:
    key = keyboard.getKey()
    mapping_inst.update(np.array([0, 0, 0]), np.array([0, 0, 0]), lidar_inst)
    if key == ord("W"):
        pass
        #go forward
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
        print(f"{mapping_inst.get_normalised(maximum_certainty_log_odds=10000)}\n\r\n\r")  # maximum_certainty_log_odds to be determined
        # print(mapping_inst.get().shape)
    loops += 1

    #localisation -> mapping -> database of map
    pass

# Enter here exit cleanup code.
