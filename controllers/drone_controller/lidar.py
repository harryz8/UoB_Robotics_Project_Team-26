import numpy as np


class Lidar:
    """
    Class that holds a Li-DAR device and its orientation and
    """

    __current_readings: np.ndarray = np.empty(0)  # initialise empty array for the readings to be put in later

    def __init__(self,
                 lidar_device,
                 axis_from_robot: tuple[int, int],
                 object_detected_given_object_prob: float,
                 empty_detected_given_empty_prob: float):
        """
        Creates the Lidar object
        :param lidar_device: The link to the drone's lidar device. This should just link to one lidar device - the one to be used by this instantiation of the Lidar class
        :param axis_from_robot: A tuple containing the drone's two axis which lie in the plane described by this lidar. The first is the axis which the Li-DAR is pointing along. Li-DARs must point along an axis. Axes are described as such: 0 - drone's x axis, 1 - drone's y axis, 2 - drone's z axis
        :param object_detected_given_object_prob: The probability that the lidar detects an object given that there really is an object for it to detect
        :param empty_detected_given_empty_prob: The probability that the lidar detects no object given that there really isn't an object for it to detect
        """
        self.device = lidar_device
        self.axis_from_robot: tuple[int, int] = axis_from_robot
        self.object_detection_accuracy = object_detected_given_object_prob
        self.empty_detection_accuracy = empty_detected_given_empty_prob

    def get_readings(self) -> np.ndarray:
        """
        Converts readings from the Li-DAR device to an np.ndarray
        :return: an np.ndarray of the readings from the Li-DAR device
        """
        return np.array(self.device.getRangeImage())

    def _get_readings_with_angle(self, robot_attitude: np.ndarray) -> np.ndarray:
        """
        Assigns all readings an angle (radians) from the map's axis which corresponds to the Li-DAR's scanning plane's first axis (self.axis_from_robot[0])
        Assumption 1: all Li-DAR readings are evenly spaced across the FOV range, i.e. there is all lidar readings have the same angle between them
        Assumption 2: all Li-DAR readings start from the leftmost when looking out from the drone along the Li-DAR's scanning plane
        :param robot_attitude: The roll, pitch and yaw of the drone (in that order). Measured in radians
        :return: readings as pairs of the distance of the reading from the drone and the angle estimated from the map's axis which corresponds to the Li-DAR's scanning plane's first axis (self.axis_from_robot[0])
        """
        # finds the drone's axis which is not one that lies in the Li-DAR's plane
        axis_not_in_lidar_plane = -1
        for axis in range(0, 3):
            if not (axis in self.axis_from_robot):
                axis_not_in_lidar_plane = axis
                break
        # returns readings as pairs of the distance of the reading from the drone and the angle estimated from the map's axis which corresponds to the Li-DAR's scanning plane's first axis (self.axis_from_robot[0])
        return np.column_stack((self.get_readings(), np.linspace(robot_attitude[axis_not_in_lidar_plane].item() - (self.device.getFov() / 2),
                                                                 robot_attitude[axis_not_in_lidar_plane].item() + (self.device.getFov() / 2),
                                                                 self.device.getHorizontalResolution())))

    def update_current_readings(self, robot_attitude: np.ndarray):
        """
        Sets the variable self.__current_readings by calculating the coordinates of the readings on the Li-DAR's scanning plane.
        :param robot_attitude: the roll, pitch and yaw of the drone (in that order). Measured in radians
        :return: None
        """
        readings = self._get_readings_with_angle(robot_attitude)  # get the readings with angle
        # Here I convert the readings that don't find any object (so are np.inf) to be past the scanning range.
        # This allows calculations to still take place and these readings will be filtered out in mapping.update because it only updates blocks in the Li-DARs scanning range (hence why I set them to something outside that range)
        inf_mask = readings[:, 0] == np.inf
        readings[:, 0][inf_mask] = 2*self.device.getMaxRange() + 1  # Put it beyond max range
        # Now I use the angle to find the components of the readings in each plane axis direction
        adj = np.cos(readings[:, 1]) * readings[:, 0]
        opp = np.sin(readings[:, 1]) * readings[:, 0]
        robot_plane_xy = np.column_stack((adj, opp))  # put them in pairs
        self.__current_readings = robot_plane_xy  # update self.__current_readings with the new readings

    @property
    def current_readings(self) -> np.ndarray:
        """
        Getter function for self.__current_readings
        :return: the current readings of each of the lidar 'rays' in terms of a position vector on the lidar plane
        """
        return self.__current_readings
