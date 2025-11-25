from mapping import *
from lidar import Lidar
import numpy as np
import sys
import time


def visually_test_mask(map_inst: Mapping, mask: np.ndarray) -> np.ndarray:
    temp_map = np.zeros_like(map_inst.get(10000))
    for index in map_inst.get_all_map_indexes()[mask]:
        temp_map[index[0], index[1], index[2]] = 1
    return temp_map


def time_function(function, *args) -> float:
    start_time = time.time()
    function(*args)
    end_time = time.time()
    return end_time - start_time


class TestDevice:
    def __init__(self, l_range_image):
        self.range_image = l_range_image
        pass

    def getFov(self):
        return np.pi

    def getHorizontalResolution(self):
        return 100

    def getMaxRange(self):
        return 1

    def getRangeImage(self):
        return self.range_image


def test_map_update(map_inst, lidar_inst):
    map_inst.update(robot_loc=np.array([0, 0, 0]), robot_attitude=np.array([0, 0, 0]), lidar_inst=lidar_inst)
    print(np.sign(map_inst.get_normalised(maximum_certainty_log_odds=10000)[:, :, 4]))
    print(map_inst.get_normalised(maximum_certainty_log_odds=10000).shape)

def test_fov_mask(map_inst, lidar_inst):
    np.set_printoptions(threshold=sys.maxsize)
    robot_loc = map_inst.prepare_map_and_update_location(np.array([0, 0, 0]), lidar_inst)
    mask = map_inst.get_lidar_fov_mask(robot_loc=robot_loc,
                                       robot_attitude=np.array([0, 0, 0]),
                                       lidar_inst=lidar_inst)
    # print(map_inst.get_all_map_indexes()[mask])
    print(visually_test_mask(map_inst, mask)[:, :, 4])
    assert map_inst.get_all_map_indexes()[mask].shape[0] == 45, "test_fov_mask() failed."

def test_initialise_blocks_in_range(robot_map):
    loc = robot_map.initialise_blocks_in_range(np.array([0, 0, 0]), 1)
    assertion = loc == np.array([4, 4, 4])
    assert assertion.all(), "test_initialise_blocks_in_range() failed."

def test_range_mask(map_inst, lidar_inst):
    robot_loc = map_inst.prepare_map_and_update_location(np.array([0, 0, 0]), lidar_inst)
    mask = map_inst.get_lidar_range_mask(robot_loc, lidar_inst)
    vals = blocks_to_meters(map_inst.get_all_map_indexes()[mask], map_inst.block_length) - robot_loc
    displacements = np.sqrt(np.sum(np.square(vals), axis=1))
    print(visually_test_mask(map_inst, mask)[:, :, 4])
    assert (displacements <= 1).all(), "test_range_mask() failed."

def test_pitch_mask(map_instance: Mapping, lidar_inst: Lidar):
    robot_loc = map_instance.prepare_map_and_update_location(np.array([0, 0, 0]), lidar_inst)
    robot_attitude = np.array([0, np.pi/2, 0])
    roll_matrix = np.array([[1, 0, 0],
                            [0, np.cos(robot_attitude[0]), -np.sin(robot_attitude[0])],
                            [0, np.sin(robot_attitude[0]), np.cos(robot_attitude[0])]])
    pitch_matrix = np.array([[np.cos(robot_attitude[1]), 0, np.sin(robot_attitude[1])],
                             [0, 1, 0],
                             [-np.sin(robot_attitude[1]), 0, np.cos(robot_attitude[1])]])
    yaw_matrix = np.array([[np.cos(robot_attitude[2]), -np.sin(robot_attitude[2]), 0],
                           [np.sin(robot_attitude[2]), np.cos(robot_attitude[2]), 0],
                           [0, 0, 1]])
    mask = map_instance.get_pitch_mask(robot_loc, roll_matrix, pitch_matrix, yaw_matrix, map_instance.get_all_map_indexes(), lidar_inst)
    print(visually_test_mask(map_instance, mask)[:, :, 4])

if __name__ == "__main__":
    range_image = 0.5 / np.cos(np.linspace(-np.pi / 2, np.pi / 2, 100))
    range_image_filter = range_image > 1
    range_image[range_image_filter] = np.inf
    mydar = Lidar(TestDevice(range_image), (0, 1), 0.9, 0.9)
    mymap = Mapping(250, 1)
    # test_initialise_blocks_in_range(mymap)
    # test_fov_mask(mymap, mydar)
    # test_range_mask(mymap, mydar)
    # print(time_function(test_map_update, mymap, mydar))
    test_pitch_mask(mymap, mydar)
    print("Everything passed")