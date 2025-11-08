import numpy as np
import math


def _xy_angle_calc(robot_loc_blocks: np.ndarray, spec_loc_blocks: tuple[int, int, int]) -> float:
    # calculates angle from x-axis for vector between these two points
    lengths = np.array(spec_loc_blocks) - robot_loc_blocks
    return np.arctan2([lengths[1]], [lengths[0]])


def _angle_calc_arr(robot_loc_meters: np.ndarray,
                    spec_loc_blocks: np.ndarray,
                    block_length_meters: float,
                    axes: tuple[int, int] = (0, 1)
                    ) -> np.ndarray:
    # calculates angle from x-axis for vector between these two points when spec_loc_blocks is an np.ndarray and robot_loc is in meters not blocks
    lengths = spec_loc_blocks - (robot_loc_meters / block_length_meters)
    return np.arctan2([lengths[:, axes[1]]], [lengths[:, axes[0]]])


def meters_to_blocks(measurement_array: np.ndarray, block_length_meters: float) -> np.ndarray:
    return measurement_array // block_length_meters


def blocks_to_meters(block_indices_array: np.ndarray, block_length_meters: float) -> np.ndarray:
    return block_indices_array * block_length_meters


def displacement_2d(start: np.ndarray, end: np.ndarray, axis: tuple[int, int]) -> np.ndarray:
    return np.sqrt(np.square(end[:, axis[0]] - start[axis[0]]) + np.square(end[:, axis[1]] - start[axis[1]]))


class Mapping:
    """
    Occupancy grid map
    """

    _map = np.ones((1, 1, 1), dtype='float64')
    origin = np.array([0, 0, 0])  # measured in blocks. Assumes the drone starts at 0 meters from home in any direction

    def __init__(self, block_length_mm: float, robot_size_blocks: int):
        self.block_length = block_length_mm / 1000
        self.robot_size_blocks = robot_size_blocks

    def _extend_to(self, coords: tuple[int, int, int]) -> np.ndarray:
        prev_start = self.origin.copy()
        map_shape = (self._map.shape - np.ones(3)).astype('i')
        self.origin = self.origin + (map_shape - np.maximum(map_shape, np.array(coords))) - (
                np.zeros(3) + np.minimum(np.zeros(3), np.array(coords)))
        # print(f"({abs(min(0, coords[0]))}, {max(0, coords[0]-map_shape[0])}), ({abs(min(0, coords[1]))}, {max(0, coords[1]-map_shape[1])}), ({abs(min(0, coords[2]))}, {max(0, coords[2]-map_shape[2])})")
        self._map = np.pad(self._map, pad_width=(
            (abs(min(0, coords[0])), max(0, coords[0] - map_shape[0])),
            (abs(min(0, coords[1])), max(0, coords[1] - map_shape[1])),
            (abs(min(0, coords[2])), max(0, coords[2] - map_shape[2]))),
                           mode='constant', constant_values=1)
        return self.origin - prev_start

    def initialise_blocks_in_range(self, robot_map_index: np.ndarray, radius: float) -> np.ndarray:
        new_blocks_max_dist: int = math.ceil(radius / self.block_length)
        new_blocks: list[tuple[int, int, int]] = [(x, y, z)
                                                  for x in range(robot_map_index[0].astype("i") - new_blocks_max_dist,
                                                                 robot_map_index[0].astype("i") + new_blocks_max_dist + 1)
                                                  for y in range(robot_map_index[1].astype("i") - new_blocks_max_dist,
                                                                 robot_map_index[1].astype("i") + new_blocks_max_dist + 1)
                                                  for z in range(robot_map_index[2].astype("i") - new_blocks_max_dist,
                                                                 robot_map_index[2].astype("i") + new_blocks_max_dist + 1)]
        change_vec = np.zeros(3)
        for new_block in new_blocks:
            if not ((new_block[0] + change_vec[0] < self._map.shape[0]) and (
                    new_block[1] + change_vec[1] < self._map.shape[1]) and (
                            new_block[2] + change_vec[2] < self._map.shape[2]) and (
                            new_block[0] + change_vec[0] >= 0) and (new_block[1] + change_vec[1] >= 0) and (
                            new_block[2] + change_vec[2] >= 0)):
                change_vec = change_vec + self._extend_to((int(new_block[0] + change_vec[0]),
                                                           int(new_block[1] + change_vec[1]),
                                                           int(new_block[2] + change_vec[2])))
        robot_map_index = (robot_map_index + change_vec).astype("i")
        return robot_map_index

    def update(self, heading_angle: float, robot_loc: np.ndarray, lidar, lidar_axis: tuple[int, int] = (0, 1)):
        """
        Update map after new lidar reading.
        
        :param lidar_axis: the axis (multiple) that define the plane in which the lidar scans
        :param lidar: the lidar to take measurements from
        :param robot_loc: an np.ndarray with all 3 measurements for the displacement along the different axis in meters
        :param heading_angle: the angle in radians between the robot and the primary lidar axis (lidar_axis[0])

        :return: None
        """
        learning_rate: int = 1

        # extend map
        robot_loc_blocks = meters_to_blocks(robot_loc, self.block_length)
        robot_loc_remainder = robot_loc % self.block_length
        robot_map_index = self.initialise_blocks_in_range(robot_map_index=robot_loc_blocks+self.origin, radius=lidar.getMaxRange())
        robot_loc_blocks = robot_map_index - self.origin
        robot_loc = blocks_to_meters(robot_loc_blocks, self.block_length) + robot_loc_remainder

        # get lidar values
        range_image_vec = np.array(lidar.getRangeImage())
        # print(f"riv: {range_image_vec.shape}")
        # print(range_image_vec)
        readings = np.column_stack((range_image_vec, np.linspace(heading_angle - (lidar.getFov() / 2),
                                                                 heading_angle + (lidar.getFov() / 2),
                                                                 lidar.getHorizontalResolution())))
        # print(f"readings: {readings}")

        # ---- update map ----
        # get all map indexes
        _map_indexes = np.array(np.meshgrid(
            np.arange(self._map.shape[0]),
            np.arange(self._map.shape[1]),
            np.arange(self._map.shape[2])
        )).T.reshape(-1, 3)
        # get angle from robot of all map indexes
        xy_angles = _angle_calc_arr(robot_loc, _map_indexes, self.block_length, axes=lidar_axis)
        # filter map indexes to get those within lidar FOV
        heading_filter = np.logical_and(xy_angles > ((heading_angle % (2 * math.pi)) - (lidar.getFov() / 2)),
                                        xy_angles < ((heading_angle % (2 * math.pi)) + (lidar.getFov() / 2)))
        learning_blocks_indices = _map_indexes[heading_filter.flatten()]
        # filter map indexes to get those within lidar range
        displacement_filter = displacement_2d(robot_loc,
                                              blocks_to_meters(learning_blocks_indices, self.block_length),
                                              axis=lidar_axis) <= lidar.getMaxRange()
        learning_blocks_indices = learning_blocks_indices[displacement_filter]
        # get xyz position of lidar readings in relation to robot
        readings_xyz = np.zeros((3, readings.shape[0]))
        # negative_angle_filter = readings[:, 1] < 0
        # fix_array = np.zeros(readings[:, 1].shape)
        # fix_array[negative_angle_filter] = 2*math.pi
        # readings[:, 1] = readings[:, 1] + fix_array
        readings_xyz[lidar_axis[0]] = (readings[:, 0] * np.cos(readings[:, 1]) + robot_loc[lidar_axis[0]])
        readings_xyz[lidar_axis[1]] = (readings[:, 0] * np.sin(readings[:, 1]) + robot_loc[lidar_axis[1]])
        # print(f"Readings: {readings_xyz}")
        for indices in learning_blocks_indices:
            # Calculate new map value for specific index
            diff = blocks_to_meters(indices, self.block_length) - readings_xyz.T
            update_val = np.sqrt(np.sum(np.square(diff), axis=0))
            # print(f"Update: {update_val}")
            too_far_filter = update_val > 0
            obstruction_filter = np.logical_and(update_val <= 0, update_val > -1)
            clear_filter = update_val <= -1
            update_val[too_far_filter] = 0
            update_val[obstruction_filter] = learning_rate / 2
            update_val[clear_filter] = - (learning_rate / 2)
            # print(f"block_dist {blocks_to_meters(indices, self.block_length)}, reading_dist {readings_xyz.T}, " +
            #       f"update_val {update_val}")
            self._map[indices[0], indices[1], indices[2]] = self._map[indices[0], indices[1], indices[2]] + np.sum(
                update_val) + 0  # <- prior = 0 for now
            # print(f"Loc {indices}, map {self._map[indices[0], indices[1], indices[2]] + np.sum(update_val) + 0}")

    def get(self):
        return self._map

    def get_coordinate(self, coord: tuple[int, int, int]) -> np.ndarray:
        """
        Returns occupancy for specified co-ordinate
        """
        return self._map[coord[0], coord[1], coord[2]]
