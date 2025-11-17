from mapping import *
from lidar import Lidar
import numpy as np
import sys


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
    map_inst.update(robot_loc=np.array([0, 0, 0]), robot_attitude=np.array([0, 0, -np.pi/2]), lidar_inst=lidar_inst)
    print(map_inst.get(maximum_certainty_log_odds=10000)[:, :, 4])
    print(map_inst.get(maximum_certainty_log_odds=10000).shape)

def test_fov_mask(map_inst, lidar_inst):
    np.set_printoptions(threshold=sys.maxsize)
    robot_loc = map_inst.prepare_map_and_update_location(np.array([0, 0, 0]), lidar_inst)
    mask = map_inst.get_lidar_fov_mask(robot_loc=robot_loc,
                                       robot_attitude=np.array([0, 0, 0]),
                                       lidar_inst=lidar_inst)
    visually_test_map(map_inst, mask, 4)
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
    visually_test_map(map_inst, mask, 4)
    assert (displacements <= 1).all(), "test_range_mask() failed."

if __name__ == "__main__":
    range_image = 0.5 / np.cos(np.linspace(-np.pi / 2, np.pi / 2, 100))
    range_image_filter = range_image > 1
    range_image[range_image_filter] = np.inf
    mydar = Lidar(TestDevice(range_image), (0, 1), 0.9, 0.9)
    mymap = Mapping(250, 1)
    # test_initialise_blocks_in_range(mymap)
    # test_fov_mask(mymap, mydar)
    # test_range_mask(mymap, mydar)
    test_map_update(mymap, mydar)
    print("Everything passed")