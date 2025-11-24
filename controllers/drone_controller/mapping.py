import threading
import numpy as np
import math, time
from lidar import Lidar


def visually_test_mask(map_inst, mask: np.ndarray) -> np.ndarray:
    temp_map = np.zeros_like(map_inst.get(10000))
    for index in mask:
        temp_map[index[0], index[1], index[2]] = 1
    return temp_map


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


def is_in_block(block_index: np.ndarray, location_meters: np.ndarray, block_length: float, axis) -> bool:
    block_index_meters = blocks_to_meters(block_index, block_length)
    return np.all((location_meters > block_index_meters) & (location_meters < (block_index_meters + block_length)), axis=axis)


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
        st_201 = time.time()
        with self._map_lock:
            retval =  np.array(np.meshgrid(
                np.arange(self.__map.shape[0]),
                np.arange(self.__map.shape[1]),
                np.arange(self.__map.shape[2])
            )).T.reshape(-1, 3)
        print(f"Get_all_map_indexes: {time.time() - st_201}")
        return retval

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
        st_101 = time.time()
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
                                       axes=(lidar_inst.axis_from_robot[0], yaw_axis))

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
        print(f"fov_mask time: {time.time() - st_101}")
        return fov_mask

    def get_lidar_range_mask(self, robot_loc: np.ndarray, lidar_inst, all_map_indexes: np.ndarray) -> np.ndarray:
        """
        Calculates a mask for keeping only map indexes which are within the range of the lidar from the robot in a list of all map indexes
        :param robot_loc: the current location of the robot in meters from the original location
        :param lidar_inst: the lidar object for the LiDAR currently in use
        :param all_map_indexes: a list of all indexes of values in the map
        :return: a mask for keeping only map indexes which are within the range of the lidar from the robot in a list of all map indexes
        """
        st_102 = time.time()
        dist_from_robot = blocks_to_meters(all_map_indexes, self.block_length) - robot_loc
        ret_val = np.sqrt(np.sum(np.square(dist_from_robot), axis=1)) <= lidar_inst.device.getMaxRange()
        print(f"get_lidar_range_mask time: {time.time() - st_102}")
        return ret_val

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
        # print("start")

        # start thread to process and get Lidar readings
        process_lidar_readings = threading.Thread(target=lidar_inst.update_current_readings, args=(robot_attitude,))  # https://www.w3schools.com/python/gloss_python_tuple_one_item.asp
        process_lidar_readings.start()

        # calculate and set the learning rates for how much we trust the readings and extend map ready for updating if necessary
        st_2 = time.time()
        learning_rate_when_object: float = np.log(lidar_inst.object_detection_accuracy/(1-lidar_inst.empty_detection_accuracy))
        learning_rate_when_empty: float = np.log((1-lidar_inst.object_detection_accuracy)/lidar_inst.empty_detection_accuracy)
        robot_loc = self.prepare_map_and_update_location(robot_loc, lidar_inst)
        print(f"map extend: {time.time() - st_2}")

        # ---- update map ----

        st_3 = time.time()
        # map_indices = self.get_all_map_indexes()

        with self._map_lock:
            square_range_plus = meters_to_blocks(
                    robot_loc+lidar_inst.device.getMaxRange(), self.block_length)
            square_range_minus = meters_to_blocks(
                robot_loc - lidar_inst.device.getMaxRange(), self.block_length)
            ranged_map_indexes =  np.array(np.meshgrid(
                np.arange(np.maximum(0, square_range_minus[0]),
                          np.minimum(self.__map.shape[0], square_range_plus[0].astype("i"))),
                np.arange(np.maximum(0, square_range_minus[1]),
                          np.minimum(self.__map.shape[1], square_range_plus[1].astype("i"))),
                np.arange(np.maximum(0, square_range_minus[2]),
                          np.minimum(self.__map.shape[2], square_range_plus[2].astype("i")))
            )).T.reshape(-1, 3).astype("i")
            # print(ranged_map_indexes)

        # filter map indexes to get those within lidar range and fov
        range_mask = self.get_lidar_range_mask(robot_loc, lidar_inst, ranged_map_indexes)
        map_indices = ranged_map_indexes[range_mask]
        fov_mask = self.get_lidar_fov_mask(robot_loc, robot_attitude, map_indices, lidar_inst)
        learning_blocks_indices = map_indices[fov_mask]

        # disallow current block
        current_index = np.where(np.all(learning_blocks_indices == meters_to_blocks(robot_loc, self.block_length), axis=1))[0]
        learning_blocks_indices = np.delete(learning_blocks_indices, current_index, axis=0)
        print(f"filtering map: {time.time() - st_3}")

        # Get the range readings from the lidar
        process_lidar_readings.join()
        readings_vec_from_robot = lidar_inst.current_readings

        st_401 = time.time()

        reading_disp = readings_vec_from_robot + robot_loc  # displacement of the lidar reading from the axes origin
        readings_dist_from_robot = np.linalg.norm(readings_vec_from_robot, axis=1)  # distance of the lidar readings from the robot
        norm = readings_vec_from_robot.T / readings_dist_from_robot.T
        block_step = norm.T * self.block_length  # a step of size 1 block in the direction of the lidar reading from robot

        # get all coords of block reading disp travels through except current
        # free_blocks_max = reading_disp - block_step
        stop_blocks = meters_to_blocks(robot_loc, self.block_length)
        block_index_meters = blocks_to_meters(stop_blocks, self.block_length)
        max_free_block = np.argmax(readings_dist_from_robot)
        looper_max = np.ceil(
            readings_dist_from_robot[max_free_block] / np.linalg.norm(block_step, axis=1)[max_free_block]
        ).astype("i")
        max_free_sub = np.zeros(shape=(looper_max, reading_disp.shape[0], reading_disp.shape[1]))
        for count in range(looper_max):
            max_free_sub[count:looper_max, :, :] -= block_step
        all_free_blocks_with_repetition = max_free_sub + reading_disp
        print(visually_test_mask(self, np.clip(meters_to_blocks(all_free_blocks_with_repetition, self.block_length), -8, 8).astype("i"))[:, :, 4])
        last_mask = np.empty(shape=all_free_blocks_with_repetition.shape[1])
        for i in range(looper_max):
            too_far_mask = is_in_block(meters_to_blocks(robot_loc, self.block_length), all_free_blocks_with_repetition[i], self.block_length, axis=1)
            print(f"1: {all_free_blocks_with_repetition[i].shape}")
            print(f"2: {(~np.logical_or(last_mask, too_far_mask)).shape}")
            all_free_blocks_with_repetition[i] = all_free_blocks_with_repetition[i][(~np.logical_or(last_mask, too_far_mask))]
            last_mask = too_far_mask
        all_free_blocks_with_repetition = all_free_blocks_with_repetition.reshape(-1, all_free_blocks_with_repetition.shape[2])
        all_free_blocks_with_repetition = meters_to_blocks(all_free_blocks_with_repetition[too_far_mask], self.block_length)

        # print(all_free_blocks_with_repetition)
        all_free_blocks, times_block_scanned = np.unique(np.array(all_free_blocks_with_repetition), return_counts=True, axis=0)
        print(all_free_blocks)
        print(visually_test_mask(self, all_free_blocks.astype("i"))[:,:,4])
        # print(np.column_stack((all_free_blocks, times_block_scanned)))
        print(f"readings_processing time: {time.time() - st_401}")

        st_1 = time.time()
        for learning_block_index in learning_blocks_indices:
            update_amount = 0

            # is the reading in the block specified by indices
            in_block = is_in_block(learning_block_index, reading_disp, self.block_length, axis=1)
            collisions = np.zeros_like(in_block) + learning_rate_when_object
            update_amount += np.sum(collisions[in_block])

            # is the block free
            times_block_scanned_index = np.where(np.all(all_free_blocks == learning_block_index, axis=1))[0]  # get num times block proven to be free
            if times_block_scanned_index.shape[0] > 0:  # check that there are lidar rays passing through block
                update_amount += times_block_scanned[times_block_scanned_index].item() * learning_rate_when_empty  # learning_rate is negative so add the info gained

            # Update map index
            with self._map_lock:
                self.__map[learning_block_index[0],
                learning_block_index[1],
                learning_block_index[2]] = self.__map[learning_block_index[0],
                learning_block_index[1],
                learning_block_index[2]] + update_amount

        print(f"update step: {time.time() - st_1}")

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