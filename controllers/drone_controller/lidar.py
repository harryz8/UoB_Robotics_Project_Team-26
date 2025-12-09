import numpy as np


class Lidar:

    __current_readings: np.ndarray = np.empty(0)  # initialise empty array

    def __init__(self,
                 lidar_device,
                 axis_from_robot: tuple[int, int],
                 object_detected_given_object_prob: float,
                 empty_detected_given_empty_prob: float):
        self.device = lidar_device
        self.axis_from_robot: tuple[int, int] = axis_from_robot
        self.object_detection_accuracy = object_detected_given_object_prob
        self.empty_detection_accuracy = empty_detected_given_empty_prob

    def get_readings(self) -> np.ndarray:
        return np.array(self.device.getRangeImage())

    def _get_readings_with_angle(self, robot_attitude: np.ndarray) -> np.ndarray:
        yaw_axis = -1
        for axis in range(0, 3):
            if not (axis in self.axis_from_robot):
                yaw_axis = axis
                break
        return np.column_stack((self.get_readings(), np.linspace(robot_attitude[yaw_axis].item() - (self.device.getFov() / 2),
                                                                 robot_attitude[yaw_axis].item() + (self.device.getFov() / 2),
                                                                 self.device.getHorizontalResolution())))

    def update_current_readings(self, robot_attitude: np.ndarray):
        # get xyz position of lidar readings in relation to robot
        readings = self._get_readings_with_angle(robot_attitude)
        inf_mask = readings[:, 0] == np.inf
        readings[:, 0][inf_mask] = 2*self.device.getMaxRange() + 1  # Put it beyond max range
        adj = np.cos(readings[:, 1]) * readings[:, 0]
        opp = np.sin(readings[:, 1]) * readings[:, 0]
        robot_plane_xy = np.column_stack((adj, opp))
        self.__current_readings = robot_plane_xy

    @property
    def current_readings(self) -> np.ndarray:
        """
        Getter function for self.__current_readings
        :return: the current readings of each of the lidar 'rays' in terms of a position vector on the lidar plane
        """
        return self.__current_readings
