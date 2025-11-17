import numpy as np
import mapping


class Lidar:

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

    def get_readings_with_angle(self, robot_attitude: np.ndarray) -> np.ndarray:
        yaw_axis = -1
        for axis in range(0, 3):
            if not (axis in self.axis_from_robot):
                yaw_axis = axis
                break
        return np.column_stack((self.get_readings(), np.linspace(robot_attitude[yaw_axis].item() - (self.device.getFov() / 2),
                                                                 robot_attitude[yaw_axis].item() + (self.device.getFov() / 2),
                                                                 self.device.getHorizontalResolution())))

    def get_readings_vector_from_robot(self, robot_attitude: np.ndarray):
        # get xyz position of lidar readings in relation to robot
        readings = self.get_readings_with_angle(robot_attitude)
        inf_mask = readings[:, 0] == np.inf
        readings[:, 0][inf_mask] = self.device.getMaxRange() + 1  # Put it beyond max range
        component_triangle_adj = np.cos(readings[:, 1]) * readings[:, 0]
        roll_triangle_hyp = np.sin(readings[:, 1]) * readings[:, 0]
        reading_angle_components = mapping.angle_in_given_plane_to_two_components(robot_attitude[1].item(),
                                                                                  roll_triangle_hyp,
                                                                                  component_triangle_adj)
        readings_xyz = np.zeros((3, readings.shape[0]))
        readings_xyz[self.axis_from_robot[0]] = (
                readings[:, 0] * np.cos(reading_angle_components[0]))
        readings_xyz[self.axis_from_robot[1]] = (
                readings[:, 0] * np.sin(reading_angle_components[0]))
        for num in range(0, 3):
            if not (num in self.axis_from_robot):
                ang = np.sin(reading_angle_components[1, :])
                ang_non_zero_mask = ang != 0.0
                dist_from_robot = np.zeros_like(ang)
                dist_from_robot[ang_non_zero_mask] = (readings[:, 0])[ang_non_zero_mask] * ang[ang_non_zero_mask]
                readings_xyz[num] = dist_from_robot
                break
        return readings_xyz.T
