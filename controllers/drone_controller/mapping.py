import threading
import numpy as np
import math
from lidar import Lidar


def _angle_calc_arr(robot_loc_meters: np.ndarray,
                    spec_loc_blocks: np.ndarray,
                    block_length_meters: float,
                    axes: tuple[int, int] = (0, 1)
                    ) -> np.ndarray:
    """
    Calculates angle for vector between these two points in the plane axes[0] by axes[1] when spec_loc_blocks is an np.ndarray and robot_loc is in meters not blocks
    :param robot_loc_meters: the x, y, z displacements of the drone from its original position. In meters
    :param spec_loc_blocks: The specified location (or block) that is the other end of the vector to which we are measuring the angle. In blocks
    :param block_length_meters: The length of one side of the blocks in meters
    :param axes: The two axis in which plane the lidar is acting, corresponding to the drone
    :return: the angle in radians
    """
    # Convert spec_loc_blocks to meters
    lengths = spec_loc_blocks - (robot_loc_meters / block_length_meters)

    # handles precision error on zeros
    precision_error_mask = np.isclose(lengths, 0)
    lengths[precision_error_mask] = 0

    # calculates the angle
    return np.arctan2([lengths[:, axes[1]]], [lengths[:, axes[0]]])


def angle_in_given_plane_to_two_components(roll_angle_radians: float,
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


def is_in_block(block_index: np.ndarray, location_meters: np.ndarray, block_length: float) -> bool:
    block_index_meters = blocks_to_meters(block_index, block_length)
    return np.all((location_meters > (block_index_meters - block_length/2)) & (location_meters < (block_index_meters + block_length/2)), axis=1)


class Mapping:
    """
    Class that builds and maintains an occupancy grid map.
    """

    prior = 0  # priors are 0

    def __init__(self, block_length_mm: float, robot_size_blocks: int, map_init_shape: tuple[int, int, int] = (1,1,1)):
        """
        Creates the Mapping object
        :param block_length_mm: the length of a side of the cube shaped blocks that the world is split into for the map
        :param robot_size_blocks: The size of the longest side of the robot in blocks
        :param map_init_shape: the size of the initial map. A balance must be struck, because too large of a map causes too much memory to be used whereas too small of a map causes lots of map copy operations to increase its size later on
        """
        # https://stackoverflow.com/questions/53026825/global-lock-causing-my-program-to-stop-running - Explains use of RLock
        self._map_lock = threading.RLock()
        self._origin_lock = threading.RLock()
        self.__map = np.zeros(map_init_shape, dtype='float32') + self.prior # Initialises the map
        self.__origin = np.array(map_init_shape) / 2  # measured in blocks. Assumes the drone starts at 0 meters from home in any direction
        self.block_length = block_length_mm / 1000
        self.robot_size_blocks = robot_size_blocks

    def _extend_to(self, coords: tuple[int, int, int]) -> np.ndarray:
        """
        Function to exchange the map with a larger copy. Requires _map_lock to already have been acquired
        :param coords: The coordinates that need to be within the new map
        :return: The vector by which the map's origin has drifted
        """
        map_shape = (self.__map.shape - np.ones(3)).astype('i')
        self.__map = np.pad(self.__map, pad_width=(
            (np.abs(np.minimum(0, coords[0])), np.maximum(0, coords[0] - map_shape[0])),
            (np.abs(np.minimum(0, coords[1])), np.maximum(0, coords[1] - map_shape[1])),
            (np.abs(np.minimum(0, coords[2])), np.maximum(0, coords[2] - map_shape[2]))),
                            mode='constant', constant_values=self.prior)
        with self._origin_lock:
            prev_start = self.__origin.copy()
            self.__origin = self.__origin + np.abs(np.minimum(0, coords))
            return self.__origin - prev_start

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
            with self._map_lock:
                if not ((new_block[0] + change_vec[0] < self.__map.shape[0]) and (
                         new_block[1] + change_vec[1] < self.__map.shape[1]) and (
                         new_block[2] + change_vec[2] < self.__map.shape[2]) and (
                         new_block[0] + change_vec[0] >= 0) and (
                         new_block[1] + change_vec[1] >= 0) and (
                         new_block[2] + change_vec[2] >= 0)):
                    change_vec = change_vec + self._extend_to((int(new_block[0] + change_vec[0]),
                                                               int(new_block[1] + change_vec[1]),
                                                               int(new_block[2] + change_vec[2])))
        robot_map_index = (robot_map_index + change_vec).astype("i")
        return robot_map_index

    def get_all_map_indexes(self) -> np.ndarray:
        with self._map_lock:
            return np.array(np.meshgrid(
                np.arange(self.__map.shape[0]),
                np.arange(self.__map.shape[1]),
                np.arange(self.__map.shape[2])
            )).T.reshape(-1, 3)

    def get_lidar_fov_mask(self,
                           robot_loc: np.ndarray,
                           robot_attitude: np.ndarray,
                           map_indexes: np.ndarray,
                           lidar_inst
                           ) -> np.ndarray:
        """
        Calculates a mask for the list of all map indexes that only keeps the ones in the FOV angle of the lidar, and on that plane
        :param robot_loc: the current location of the robot in meters
        :param robot_attitude: the roll, pitch and yaw of the robot in radians
        :param map_indexes: the list of all map indexes
        :param lidar_inst: the Lidar object you're referring to
        :return: a mask for the list of all map indexes that only keeps the ones in the FOV angle, and on that plane
        """
        # get FOV angles
        component_triangle_adj = np.cos(lidar_inst.device.getFov() / 2) * lidar_inst.device.getMaxRange()
        roll_triangle_hyp = np.sin(lidar_inst.device.getFov() / 2) * lidar_inst.device.getMaxRange()
        fov_components: np.ndarray = angle_in_given_plane_to_two_components(robot_attitude[lidar_inst.axis_from_robot[0]].item(),
                                                                            roll_triangle_hyp,
                                                                            component_triangle_adj)

        # get pitch and yaw from robot of all map indexes
        yaw_axis = -1
        for axis in range(0, 3):
            if not (axis in lidar_inst.axis_from_robot):
                yaw_axis = axis
                break
        yaw_angles = _angle_calc_arr(robot_loc, map_indexes, self.block_length, axes=lidar_inst.axis_from_robot)
        pitch_angles = _angle_calc_arr(robot_loc, map_indexes, self.block_length,
                                       axes=(lidar_inst.axis_from_robot[0],
                                             yaw_axis))

        # calculate filter
        if (robot_attitude[yaw_axis].item() - fov_components[0].item()) % (2*np.pi) > (
                robot_attitude[yaw_axis].item() + fov_components[0].item()) % (2*np.pi):
            yaw_filter = np.logical_or(
                (yaw_angles % (2*np.pi)) >= (
                        (robot_attitude[yaw_axis].item() - fov_components[0].item()) % (2*np.pi)),
                (yaw_angles % (2*np.pi)) <= (
                        (robot_attitude[yaw_axis].item() + fov_components[0].item()) % (2*np.pi)))
        else:
            yaw_filter = np.logical_and(
                (yaw_angles % (2 * np.pi)) >= (
                            (robot_attitude[yaw_axis].item() - fov_components[0].item()) % (2 * np.pi)),
                (yaw_angles % (2 * np.pi)) <= (
                            (robot_attitude[yaw_axis].item() + fov_components[0].item()) % (2 * np.pi)))

        if (robot_attitude[lidar_inst.axis_from_robot[1]].item() - fov_components[1].item()) % (2*np.pi) > (robot_attitude[lidar_inst.axis_from_robot[1]].item() + fov_components[1].item()) % (2*np.pi):
            pitch_filter = np.logical_or(
                (pitch_angles % (2*np.pi)) >= ((robot_attitude[lidar_inst.axis_from_robot[1]].item() - fov_components[1].item()) % (2*np.pi)),
                (pitch_angles % (2*np.pi)) <= ((robot_attitude[lidar_inst.axis_from_robot[1]].item() + fov_components[1].item()) % (2*np.pi)))
        else:
            pitch_filter = np.logical_and(
                (pitch_angles % np.pi) >= ((robot_attitude[lidar_inst.axis_from_robot[1]].item() - fov_components[1].item()) % np.pi),
                (pitch_angles % np.pi) <= ((robot_attitude[lidar_inst.axis_from_robot[1]].item() + fov_components[1].item()) % np.pi))

        # filter map indexes to get those within lidar FOV
        fov_mask = yaw_filter.flatten() & pitch_filter.flatten()
        return fov_mask

    def get_lidar_range_mask(self, robot_loc: np.ndarray, lidar_inst, all_map_indexes: np.ndarray) -> np.ndarray:
        dist_from_robot = blocks_to_meters(all_map_indexes, self.block_length) - robot_loc
        return np.sqrt(np.sum(np.square(dist_from_robot), axis=1)) <= 1

    def prepare_map_and_update_location(self, robot_loc: np.ndarray, lidar_inst) -> np.ndarray:
        # extend map
        robot_loc_blocks = meters_to_blocks(robot_loc, self.block_length)
        robot_loc_remainder = robot_loc % self.block_length
        with self._origin_lock:
            robot_map_index = self.initialise_blocks_in_range(robot_map_index=robot_loc_blocks + self.__origin,
                                                          radius=lidar_inst.device.getMaxRange())
        robot_loc_blocks = robot_map_index  # - self.origin
        robot_loc = blocks_to_meters(robot_loc_blocks, self.block_length) + robot_loc_remainder
        return robot_loc

    def update(self, robot_loc: np.ndarray, robot_attitude: np.ndarray, lidar_inst: Lidar):
        """
        Update map after new lidar reading.

        :param lidar_inst: the Lidar object to take measurements from
        :param robot_loc: an np.ndarray with all 3 measurements for the displacement along the different axis in meters.
                          In the order x, y, z
        :param robot_attitude: The roll, pitch and yaw of the robot (in that order). Measured in radians
        
        :return: None
        """

        # start thread to process and get Lidar readings
        process_lidar_readings = threading.Thread(target=lidar_inst.update_current_readings, args=(robot_attitude,))  # https://www.w3schools.com/python/gloss_python_tuple_one_item.asp
        process_lidar_readings.start()

        # calculate and set the learning rates for how much we trust the readings and extend map ready for updating if necessary
        learning_rate_when_object: float = np.log(lidar_inst.object_detection_accuracy/(1-lidar_inst.empty_detection_accuracy))
        learning_rate_when_empty: float = np.log((1-lidar_inst.object_detection_accuracy)/lidar_inst.empty_detection_accuracy)
        robot_loc = self.prepare_map_and_update_location(robot_loc, lidar_inst)

        # ---- update map ----

        all_map_indexes = self.get_all_map_indexes()

        # filter map indexes to get those within lidar range and fov
        fov_mask = self.get_lidar_fov_mask(robot_loc, robot_attitude, all_map_indexes, lidar_inst)
        range_mask = self.get_lidar_range_mask(robot_loc, lidar_inst, all_map_indexes)
        learning_blocks_indices = all_map_indexes[fov_mask & range_mask]

        # disallow current block
        current_index = np.where(np.all(learning_blocks_indices == meters_to_blocks(robot_loc, self.block_length), axis=1))[0]
        learning_blocks_indices = np.delete(learning_blocks_indices, current_index, axis=0)

        # Get the range readings from the lidar
        process_lidar_readings.join()
        readings_vec_from_robot = lidar_inst.current_readings

        # i = 0
        for indices in learning_blocks_indices:
            update_amount = 0

            # is the reading in the block specified by indices
            reading_disp = readings_vec_from_robot + robot_loc
            reading_dist = np.linalg.norm(reading_disp, axis=1)
            in_block = is_in_block(indices, reading_disp, self.block_length)
            collisions = np.zeros_like(in_block) + learning_rate_when_object
            update_amount += np.sum(collisions[in_block])

            # is the block free
            norm = reading_disp.T / reading_dist.T
            block_step = norm.T * self.block_length
            cur_disp = reading_disp - block_step
            max_index = np.argmax(reading_dist)
            while np.sum(np.sign(cur_disp[max_index])) * reading_dist[max_index] > 0:
                in_block = is_in_block(indices, cur_disp, self.block_length)
                collisions = np.zeros_like(in_block) + learning_rate_when_empty
                over_mask = np.sum(np.sign(cur_disp), axis=1) * reading_dist < 0
                collisions[over_mask] = 0
                update_amount += np.sum(collisions[in_block])
                cur_disp -= block_step

            # Update map index
            with self._map_lock:
                self.__map[indices[0], indices[1], indices[2]] = self.__map[indices[0], indices[1], indices[2]] + update_amount

    def get(self, maximum_certainty_log_odds: float) -> np.ndarray:
        # Returns a copy of the map where the certainty is limited to [-self.max_certainty, self.max_certainty] so that no one area becomes overly important making all other areas of the map relatively negligible
        # This could have happened, for example, when the drone is stopped at one location for a long time.
        with self._map_lock:
            map_copy = self.__map.copy()
            max_filter = self.__map > maximum_certainty_log_odds
            min_filter = self.__map < -maximum_certainty_log_odds
        map_copy[max_filter] = maximum_certainty_log_odds
        map_copy[min_filter] = -maximum_certainty_log_odds
        return map_copy

    def get_normalised(self, maximum_certainty_log_odds: float) -> np.ndarray:
        """
        Get the occupancy map where all values range between -1 and 1 inclusive
        :param maximum_certainty_log_odds: the most outlying value for all log_odds to be scaled to. No value will be allowed to be greater that it or less than its negative.
        :return: the occupancy map where all values range between -1 and 1 inclusive
        """
        limit_map = self.get(maximum_certainty_log_odds)
        map_copy = limit_map.copy()
        return map_copy / maximum_certainty_log_odds

    def __getitem__(self, key: tuple[int, int, int]) -> np.ndarray:
        """
        Returns occupancy for specified co-ordinate
        :param key: tuple[int, int, int] : the specified co-ordinate
        :return: the occupancy log odd for specified co-ordinate
        """
        with self._map_lock:
            return self.__map[key[0], key[1], key[2]]

    @property
    def origin(self) -> np.ndarray:
        with self._origin_lock:
            return self.__origin

    def __str__(self):
        with self._map_lock:
            return str(self.__map)