import threading
import numpy as np
import math
import matplotlib.pyplot as plt
from lidar import Lidar


def fix_zero_precision(array: np.ndarray) -> np.ndarray:
    """
    NumPy and python as a whole has precision errors when doing operations on floats.
    This causes the most errors when checking for zeros as they could be slightly off 0.
    NumPy provides a function to handle this `np.isclose()` and I have used this to set values that should be 0 back to 0.
    :param array: The array in which to change elements back to 0
    :return: the array but with elements that should be 0 set to 0
    """
    precision_error_mask = np.isclose(array, 0)
    array[precision_error_mask] = 0
    return array


def meters_to_blocks(measurement_array: np.ndarray, block_length_meters: float) -> np.ndarray:
    return measurement_array // block_length_meters


def blocks_to_meters(block_indices_array: np.ndarray, block_length_meters: float) -> np.ndarray:
    """
    Calculates the displacements in each dimension of the centre of each block indexed by block_indices_array from the map origin
    :param block_indices_array: a list of lists of indexes for each dimension of map, each list indexing one block
    :param block_length_meters: the length of one size of a block in meters
    :return: an np.ndarray containing a list of dimension many displacements for each block indexed by block_indices_array
    """
    return block_indices_array * block_length_meters + block_length_meters/2


def is_in_block(block_index: np.ndarray, location_meters: np.ndarray, block_length: float) -> bool:
    block_index_meters = blocks_to_meters(block_index, block_length)
    return np.all((location_meters > (block_index_meters - block_length/2)) & (location_meters < (block_index_meters + block_length/2)), axis=1)


def get_lidar_plane_axes_in_terms_of_map_axes(roll_matrix: np.ndarray,
                                       pitch_matrix: np.ndarray,
                                       yaw_matrix: np.ndarray):
    # prepare plane axes as if robot had roll, pitch, yaw all = 0
    robot_lidar_plane_axes_in_terms_of_map_axes = np.identity(3)
    # get plane axes under robot roll, pitch, yaw
    rotate_all = np.matmul(yaw_matrix, np.matmul(roll_matrix, pitch_matrix))
    robot_lidar_plane_axes_in_terms_of_map_axes = fix_zero_precision(np.matmul(robot_lidar_plane_axes_in_terms_of_map_axes, rotate_all))
    return robot_lidar_plane_axes_in_terms_of_map_axes.copy()


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
        self.__origin = np.array(map_init_shape) // 2  # measured in blocks. Assumes the drone starts at 0 meters from home in any direction
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
        """
        Creates a list of indexes to every block currently in the map
        :return: that list as an np.ndarray
        """
        with self._map_lock:
            return np.array(np.meshgrid(
                np.arange(self.__map.shape[0]),
                np.arange(self.__map.shape[1]),
                np.arange(self.__map.shape[2])
            )).T.reshape(-1, 3)

    def restrict_map_indexes_to_within_fov(self, robot_loc: np.ndarray,
                                           lidar_plane_axes: np.ndarray,
                                           all_map_indexes: np.ndarray,
                                           lidar_inst: Lidar) -> np.ndarray:
        """
        Takes a list of map indexes and filters out ones that index map blocks which the lidar does not scan in
        :param robot_loc: the location of the robot from the map origin, in meters
        :param lidar_plane_axes: vectors describing the direction of the lidar_plane's axes in terms of the map's axis
        :param all_map_indexes: a list of map indexes to be filtered
        :param lidar_inst: the Lidar object for the lidar which is scanning the map blocks whose indexes we are wanting
        :return: a list of map indexes (np.ndarray) for the blocks in the map which are at least somewhat scanned by the lidar_inst
        """
        yaw_axis = -1
        for axis in range(0, 3):
            if not (axis in lidar_inst.axis_from_robot):
                yaw_axis = axis
                break
        lidar_plane_normal = lidar_plane_axes[yaw_axis]
        # get the displacement (m) of all map_indexes from the origin of the axes
        disp_from_axes = (all_map_indexes * self.block_length) + (self.block_length / 2)
        # solve equation distance between plane and line perpendicular to plane that passes through the point disp_from_axes
        # nr = c where c is the dot of n and a point on the plane, n is the normal vec of the pane and r is the equation of the line
        # then solve for the lambda that is the equation of the line
        c = np.dot(lidar_plane_normal, robot_loc)
        n_square = np.dot(lidar_plane_normal, lidar_plane_normal)
        n_point = np.dot(disp_from_axes, lidar_plane_normal)
        rhs = c - n_point
        lamb = rhs / n_square
        # get displacement of parallel line from plane
        shortest_disp_from_plane = lamb.reshape(lamb.shape[0], 1) @ lidar_plane_normal.reshape(1, lidar_plane_normal.shape[0])
        in_block_mask = np.all(np.logical_and(
            shortest_disp_from_plane >= -self.block_length / 2,
            shortest_disp_from_plane < self.block_length / 2
        ), axis=1)
        filtered_indexes = disp_from_axes[in_block_mask]
        coords_on_plane = filtered_indexes @ lidar_plane_axes
        plane_robot_loc = robot_loc @ lidar_plane_axes
        coords_on_plane_from_robot = coords_on_plane - plane_robot_loc
        angle_from_lidar_1_axis = np.arctan2([coords_on_plane_from_robot[:, lidar_inst.axis_from_robot[1]]],
                                             [coords_on_plane_from_robot[:, lidar_inst.axis_from_robot[0]]])
        fov_angle_mask = np.logical_and(angle_from_lidar_1_axis > -lidar_inst.device.getFov()/2,
                                        angle_from_lidar_1_axis < lidar_inst.device.getFov()/2)
        return all_map_indexes[in_block_mask][fov_angle_mask.flatten()]

    def get_lidar_range_mask(self, robot_loc: np.ndarray, lidar_inst, all_map_indexes: np.ndarray) -> np.ndarray:
        """
        Calculates a mask for keeping only map indexes which are within the range of the lidar from the robot in a list of all map indexes
        :param robot_loc: the current location of the robot in meters from the original location
        :param lidar_inst: the lidar object for the LiDAR currently in use
        :param all_map_indexes: a list of all indexes of values in the map
        :return: a mask for keeping only map indexes which are within the range of the lidar from the robot in a list of all map indexes
        """
        dist_from_robot = blocks_to_meters(all_map_indexes, self.block_length) - robot_loc
        return np.sqrt(np.sum(np.square(dist_from_robot), axis=1)) <= lidar_inst.device.getMaxRange()

    def prepare_map_and_update_location(self, robot_loc: np.ndarray, lidar_inst) -> np.ndarray:
        # extend map
        robot_loc_blocks = meters_to_blocks(robot_loc, self.block_length)
        robot_loc_remainder = robot_loc % self.block_length
        with self._origin_lock:
            robot_map_index = self.initialise_blocks_in_range(robot_map_index=robot_loc_blocks + self.__origin,
                                                          radius=lidar_inst.device.getMaxRange())
        robot_loc_blocks = robot_map_index  # - self.origin
        robot_loc = blocks_to_meters(robot_loc_blocks, self.block_length) + robot_loc_remainder - self.block_length/2
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

        # Calculate the rotation matrices for pitch roll and yaw
        roll_matrix = np.array([[1, 0, 0],
                                [0, np.cos(robot_attitude[0]), -np.sin(robot_attitude[0])],
                                [0, np.sin(robot_attitude[0]), np.cos(robot_attitude[0])]])
        pitch_matrix = np.array([[np.cos(robot_attitude[1]), 0, np.sin(robot_attitude[1])],
                                 [0, 1, 0],
                                 [-np.sin(robot_attitude[1]), 0, np.cos(robot_attitude[1])]])
        yaw_matrix = np.array([[np.cos(robot_attitude[2]), -np.sin(robot_attitude[2]), 0],
                               [np.sin(robot_attitude[2]), np.cos(robot_attitude[2]), 0],
                               [0, 0, 1]])

        # calculate and set the learning rates for how much we trust the readings and extend map ready for updating if necessary
        learning_rate_when_object: float = np.log(lidar_inst.object_detection_accuracy/(1-lidar_inst.empty_detection_accuracy))
        learning_rate_when_empty: float = np.log((1-lidar_inst.object_detection_accuracy)/lidar_inst.empty_detection_accuracy)
        robot_loc = robot_loc + self.block_length/2
        robot_loc = self.prepare_map_and_update_location(robot_loc, lidar_inst)

        # ---- update map ----

        # get all map indexes in a square around the robot with minimum distance from the robot of the lidar max range
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

        # filter map indexes to get those within lidar range and fov
        range_mask = self.get_lidar_range_mask(robot_loc, lidar_inst, ranged_map_indexes)
        all_map_indexes = ranged_map_indexes[range_mask]
        lidar_plane_axes = get_lidar_plane_axes_in_terms_of_map_axes(roll_matrix, pitch_matrix, yaw_matrix)
        learning_blocks_indices = self.restrict_map_indexes_to_within_fov(robot_loc, lidar_plane_axes, all_map_indexes, lidar_inst)

        # disallow current block
        current_index = np.where(np.all(learning_blocks_indices == meters_to_blocks(robot_loc, self.block_length), axis=1))[0]
        learning_blocks_indices = np.delete(learning_blocks_indices, current_index, axis=0)

        # learning_blocks_indices = self.get_all_map_indexes()  # TODO: REMOVE

        # Get the range readings from the lidar
        process_lidar_readings.join()
        readings_plane_xy = lidar_inst.current_readings
        readings_vec_from_robot = (np.tile(lidar_plane_axes[lidar_inst.axis_from_robot[0]], (readings_plane_xy.shape[0], 1)).T*readings_plane_xy[:, 0] + np.tile(lidar_plane_axes[lidar_inst.axis_from_robot[1]], (readings_plane_xy.shape[0], 1)).T*readings_plane_xy[:, 1]).T
        reading_disp = readings_vec_from_robot + robot_loc
        reading_disp = reading_disp.astype(np.float32)
        reading_dist_from_robot = np.linalg.norm(readings_vec_from_robot, axis=1)

        # i = 0
        for indices in learning_blocks_indices:
            update_amount = 0

            # is the reading in the block specified by indices
            in_block = is_in_block(indices, reading_disp, self.block_length)
            collisions = np.zeros_like(in_block) + learning_rate_when_object
            update_amount += np.sum(collisions[in_block])

            # is the block free
            norm = readings_vec_from_robot.T / reading_dist_from_robot.T
            block_step = norm.T * self.block_length
            cur_disp = robot_loc + block_step
            max_index = np.argmax(reading_dist_from_robot)
            max_loops = reading_dist_from_robot[max_index] / np.linalg.norm(block_step[max_index])
            for _ in range(max_loops.astype("i")):  # -1
                in_block = is_in_block(indices, cur_disp, self.block_length)
                collisions = np.zeros_like(in_block) + learning_rate_when_empty
                over_mask = np.linalg.norm(cur_disp - robot_loc, axis=1)//np.linalg.norm(block_step, axis=1) >= reading_dist_from_robot//np.linalg.norm(block_step, axis=1)
                collisions[over_mask] = 0
                update_amount += np.sum(collisions[in_block])
                cur_disp += block_step

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

    def get_visual_map(self, axis, index):
        """
        Creates an image from any 2D slice of the map
        :param axis: the axis to index
        :param index: the index for that axis
        :return: none
        """
        plt.figure(figsize=(10, 10))
        plt.axis('off')
        slicer = [slice(None), slice(None), slice(None)]
        slicer[axis] = index
        plt.imshow(np.negative(self.get_normalised(maximum_certainty_log_odds=10000))[tuple(slicer)], cmap='gray')
        plt.show()

    def save_map(self, filename = "map.txt"):
        """
        Saves map to requested filename in format in which numpy prints maps
        :param filename: the text file in which to save the map
        :return: none
        """
        with open(filename, "w") as map_file:
            map_file.write(str(self))

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