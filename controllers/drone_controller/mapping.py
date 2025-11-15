import numpy as np
import math
from lidar import Lidar


def _angle_calc_arr(robot_loc_meters: np.ndarray,
                    spec_loc_blocks: np.ndarray,
                    block_length_meters: float,
                    axes: tuple[int, int] = (0, 1)
                    ) -> np.ndarray:
    # calculates angle from x-axis for vector between these two points when spec_loc_blocks is an np.ndarray and robot_loc is in meters not blocks
    lengths = spec_loc_blocks - (robot_loc_meters / block_length_meters)
    return np.arctan2([lengths[:, axes[1]]], [lengths[:, axes[0]]])


def angle_in_given_plane_to_two_components(
        roll_angle_radians: float,
        roll_triangle_hyp: np.ndarray,
        component_triangle_adj: np.ndarray
) -> np.ndarray:
    first_roll_triangle_adj = roll_triangle_hyp * np.cos(roll_angle_radians)
    ang = np.round(np.cos(np.pi / 2 - roll_angle_radians), 15)
    if ang == 0.0:
        second_roll_triangle_adj = 0
    else:
        second_roll_triangle_adj = roll_triangle_hyp * np.round(np.cos(np.pi / 2 - roll_angle_radians), 15)
    if isinstance(second_roll_triangle_adj, np.ndarray):
        nan_filter = np.isnan(second_roll_triangle_adj)
        second_roll_triangle_adj[nan_filter] = 0
    else:
        second_roll_triangle_adj = 0 if math.isnan(second_roll_triangle_adj) else second_roll_triangle_adj
    return np.array([np.arctan2(first_roll_triangle_adj, component_triangle_adj),
                     np.arctan2(second_roll_triangle_adj, component_triangle_adj)])


def meters_to_blocks(measurement_array: np.ndarray, block_length_meters: float) -> np.ndarray:
    return measurement_array // block_length_meters


def blocks_to_meters(block_indices_array: np.ndarray, block_length_meters: float) -> np.ndarray:
    return block_indices_array * block_length_meters


# def displacement_2d(start: np.ndarray, end: np.ndarray, axis: tuple[int, int]) -> np.ndarray:
#     print(f"end : {end}")
#     print(f"start : {start}")
#     return np.sqrt(np.square(end[:, axis[0]] - start[axis[0]]) + np.square(end[:, axis[1]] - start[axis[1]]))


class Mapping:
    """
    Occupancy grid map
    """

    prior = 0  # priors are 0
    _map = np.zeros((1, 1, 1), dtype='float64') + prior
    origin = np.array([0, 0, 0])  # measured in blocks. Assumes the drone starts at 0 meters from home in any direction

    def __init__(self, block_length_mm: float, robot_size_blocks: int, maximum_certainty_log_odds: float):
        self.block_length = block_length_mm / 1000
        self.robot_size_blocks = robot_size_blocks
        self.max_certainty = maximum_certainty_log_odds

    def _extend_to(self, coords: tuple[int, int, int]) -> np.ndarray:
        prev_start = self.origin.copy()
        map_shape = (self._map.shape - np.ones(3)).astype('i')
        self.origin = self.origin + np.abs(np.minimum(0, coords))
        self._map = np.pad(self._map, pad_width=(
            (np.abs(np.minimum(0, coords[0])), np.maximum(0, coords[0] - map_shape[0])),
            (np.abs(np.minimum(0, coords[1])), np.maximum(0, coords[1] - map_shape[1])),
            (np.abs(np.minimum(0, coords[2])), np.maximum(0, coords[2] - map_shape[2]))),
                           mode='constant', constant_values=self.prior)
        return self.origin - prev_start

    def initialise_blocks_in_range(self, robot_map_index: np.ndarray, radius: float) -> np.ndarray:
        new_blocks_max_dist: int = math.ceil(radius / self.block_length)
        new_blocks: list[tuple[int, int, int]] = [(x, y, z)
                                                  for x in range(robot_map_index[0].astype("i") - new_blocks_max_dist,
                                                                 robot_map_index[0].astype(
                                                                     "i") + new_blocks_max_dist + 1)
                                                  for y in range(robot_map_index[1].astype("i") - new_blocks_max_dist,
                                                                 robot_map_index[1].astype(
                                                                     "i") + new_blocks_max_dist + 1)
                                                  for z in range(robot_map_index[2].astype("i") - new_blocks_max_dist,
                                                                 robot_map_index[2].astype(
                                                                     "i") + new_blocks_max_dist + 1)]
        change_vec = np.zeros(3)
        for new_block in [min(new_blocks), max(new_blocks)]:
            if not ((new_block[0] + change_vec[0] < self._map.shape[0]) and (
                     new_block[1] + change_vec[1] < self._map.shape[1]) and (
                     new_block[2] + change_vec[2] < self._map.shape[2]) and (
                     new_block[0] + change_vec[0] >= 0) and (
                     new_block[1] + change_vec[1] >= 0) and (
                     new_block[2] + change_vec[2] >= 0)):
                change_vec = change_vec + self._extend_to((int(new_block[0] + change_vec[0]),
                                                           int(new_block[1] + change_vec[1]),
                                                           int(new_block[2] + change_vec[2])))
        robot_map_index = (robot_map_index + change_vec).astype("i")
        return robot_map_index

    def get_all_map_indexes(self) -> np.ndarray:
        return np.array(np.meshgrid(
            np.arange(self._map.shape[0]),
            np.arange(self._map.shape[1]),
            np.arange(self._map.shape[2])
        )).T.reshape(-1, 3)

    def get_lidar_fov_mask(self,
                           robot_loc: np.ndarray,
                           robot_attitude: np.ndarray,
                           lidar_inst
                           ) -> np.ndarray:
        map_indexes = self.get_all_map_indexes()

        # get FOV angles
        component_triangle_adj = np.cos(lidar_inst.device.getFov() / 2) * lidar_inst.device.getMaxRange()
        roll_triangle_hyp = np.sin(lidar_inst.device.getFov() / 2) * lidar_inst.device.getMaxRange()
        fov_components: np.ndarray = angle_in_given_plane_to_two_components(robot_attitude[1].item(),
                                                                            roll_triangle_hyp,
                                                                            component_triangle_adj)

        # get pitch and yaw from robot of all map indexes
        yaw_angles = _angle_calc_arr(robot_loc, map_indexes, self.block_length, axes=lidar_inst.axis_from_robot)
        pitch_angles = _angle_calc_arr(robot_loc, map_indexes, self.block_length,
                                       axes=(lidar_inst.axis_from_robot[0],
                                             (lidar_inst.axis_from_robot[1] + 1) % 3))

        # calculate filter
        if (robot_attitude[2].item() - fov_components[0].item()) % (2*np.pi) > (robot_attitude[2].item() + fov_components[0].item()) % (2*np.pi):
            yaw_filter = np.logical_or(
                (yaw_angles % (2*np.pi)) >= (
                        (robot_attitude[2].item() - fov_components[0].item()) % (2*np.pi)),
                (yaw_angles % (2*np.pi)) <= (
                        (robot_attitude[2].item() + fov_components[0].item()) % (2*np.pi)))
        else:
            yaw_filter = np.logical_and(
                (yaw_angles % (2 * np.pi)) >= (
                            (robot_attitude[2].item() - fov_components[0].item()) % (2 * np.pi)),
                (yaw_angles % (2 * np.pi)) <= (
                            (robot_attitude[2].item() + fov_components[0].item()) % (2 * np.pi)))

        if (robot_attitude[0].item() - fov_components[1].item()) % (2*np.pi) > (robot_attitude[0].item() + fov_components[1].item()) % (2*np.pi):
            pitch_filter = np.logical_or(
                (pitch_angles % (2*np.pi)) >= ((robot_attitude[0].item() - fov_components[1].item()) % (2*np.pi)),
                (pitch_angles % (2*np.pi)) <= ((robot_attitude[0].item() + fov_components[1].item()) % (2*np.pi)))
        else:
            pitch_filter = np.logical_and(
                (pitch_angles % np.pi) >= ((robot_attitude[0].item() - fov_components[1].item()) % np.pi),
                (pitch_angles % np.pi) <= ((robot_attitude[0].item() + fov_components[1].item()) % np.pi))

        # filter map indexes to get those within lidar FOV
        fov_mask = yaw_filter.flatten() & pitch_filter.flatten()
        return fov_mask

    def get_lidar_range_mask(self, robot_loc: np.ndarray, lidar_inst):
        dist_from_robot = blocks_to_meters(self.get_all_map_indexes(), self.block_length) - robot_loc
        return np.sqrt(np.sum(np.square(dist_from_robot), axis=1)) <= 1

    def prepare_map_and_update_location(self, robot_loc: np.ndarray, lidar_inst) -> np.ndarray:
        # extend map
        robot_loc_blocks = meters_to_blocks(robot_loc, self.block_length)
        robot_loc_remainder = robot_loc % self.block_length
        robot_map_index = self.initialise_blocks_in_range(robot_map_index=robot_loc_blocks + self.origin,
                                                          radius=lidar_inst.device.getMaxRange())
        robot_loc_blocks = robot_map_index  # - self.origin
        robot_loc = blocks_to_meters(robot_loc_blocks, self.block_length) + robot_loc_remainder
        return robot_loc

    def is_in_block(self, block_index: np.ndarray, location_meters: np.ndarray) -> bool:
        block_index_meters = blocks_to_meters(block_index, self.block_length)
        return np.all((location_meters > (block_index_meters - self.block_length/2)) & (location_meters < (block_index_meters + self.block_length/2)), axis=1)

    def update(self, robot_loc: np.ndarray, robot_attitude: np.ndarray, lidar_inst: Lidar):
        """
        Update map after new lidar reading.

        :param lidar_inst: the Lidar object to take measurements from
        :param robot_loc: an np.ndarray with all 3 measurements for the displacement along the different axis in meters.
                          In the order x, y, z
        :param robot_attitude: The pitch, roll and yaw of the robot (in that order). Measured in radians
        
        :return: None
        """
        learning_rate_when_object: float = np.log(lidar_inst.object_detection_accuracy/(1-lidar_inst.empty_detection_accuracy))
        learning_rate_when_empty: float = np.log((1-lidar_inst.object_detection_accuracy)/lidar_inst.empty_detection_accuracy)

        robot_loc = self.prepare_map_and_update_location(robot_loc, lidar_inst)

        # ---- update map ----

        # filter map indexes to get those within lidar range and fov
        fov_mask = self.get_lidar_fov_mask(robot_loc, robot_attitude, lidar_inst)
        range_mask = self.get_lidar_range_mask(robot_loc, lidar_inst)
        learning_blocks_indices = self.get_all_map_indexes()[fov_mask & range_mask]

        # disallow current block
        current_index = np.where(np.all(learning_blocks_indices == meters_to_blocks(robot_loc, self.block_length), axis=1))[0]
        learning_blocks_indices = np.delete(learning_blocks_indices, current_index, axis=0)

        # Get the range readings from the lidar
        readings_vec_from_robot = lidar_inst.get_readings_vector_from_robot(robot_attitude)

        # i = 0
        for indices in learning_blocks_indices:
            update_amount = 0

            # is the reading in the block specified by indices
            reading_disp = readings_vec_from_robot + robot_loc
            reading_dist = np.linalg.norm(reading_disp, axis=1)
            in_block = self.is_in_block(indices, reading_disp)
            collisions = np.zeros_like(in_block) + learning_rate_when_object
            update_amount += np.sum(collisions[in_block])

            # is the block free
            norm = reading_disp.T / reading_dist.T
            block_step = norm.T * self.block_length
            cur_disp = reading_disp - block_step
            max_index = np.argmax(reading_dist)
            while np.sum(np.sign(cur_disp[max_index])) * reading_dist[max_index] > 0:
                in_block = self.is_in_block(indices, cur_disp)
                collisions = np.zeros_like(in_block) + learning_rate_when_empty
                over_mask = np.sum(np.sign(cur_disp), axis=1) * reading_dist < 0
                collisions[over_mask] = 0
                update_amount += np.sum(collisions[in_block])
                cur_disp -= block_step

            # Update map index
            self._map[indices[0], indices[1], indices[2]] = self._map[indices[0], indices[1], indices[2]] + update_amount

    def get(self) -> np.ndarray:
        # Returns a copy of the map where the certainty is limited to [-self.max_certainty, self.max_certainty] so that no one area becomes overly important making all other areas of the map relatively negligible
        # This could have happened, for example, when the drone is stopped at one location for a long time.
        map_copy = self._map.copy()
        max_filter = self._map > self.max_certainty
        min_filter = self._map < -self.max_certainty
        map_copy[max_filter] = self.max_certainty
        map_copy[min_filter] = -self.max_certainty
        return map_copy

    def get_coordinate(self, coord: tuple[int, int, int]) -> np.ndarray:
        """
        Returns occupancy for specified co-ordinate
        """
        return self._map[coord[0], coord[1], coord[2]]
