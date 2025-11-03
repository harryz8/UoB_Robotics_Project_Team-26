import numpy as np


def _xy_angle_calc(robot_loc_blocks: tuple[int, int, int], spec_loc_blocks: tuple[int, int, int]):
    # calculates angle from x-axis for vector between these two points
    lengths = np.array(spec_loc_blocks) - np.array(robot_loc_blocks)
    return np.arctan([lengths[1] / lengths[0]])


class Mapping:
    """
    Occupancy grid map
    """

    _map = np.ones((1, 1, 1))
    start_cell = np.array([0, 0, 0])

    def __init__(self, block_length_mm: float, robot_size_blocks: int, lidar):
        self.lidar = lidar
        self.block_length = block_length_mm / 1000
        self.field_of_view = lidar.getFov()
        self.robot_size_blocks = robot_size_blocks

    def _extend_to(self, coords: tuple[int, int, int]):
        prev_start = self.start_cell.copy()
        map_shape = self._map.shape
        self.start_cell = self.start_cell + 
                            (np.array(self._map.shape) - np.maximum(np.array(map_shape), np.array(coords))) - 
                            (np.zeros(3) + np.minimum(np.zeros(3), np.array(coords)))
        self._map = np.pad(self._map, pad_width=(
        (abs(min(0, coords[0])), max(map_shape[0], coords[0])),
        (abs(min(0, coords[1])), max(map_shape[1], coords[1])),
        (abs(min(0, coords[2])), max(map_shape[2], coords[2]))), 
        mode = 'constant', constant_values = 1)
        return self.start_cell - prev_start

    def update(self, xy_heading: tuple[int, int], robot_loc: tuple[int, int, int]):
        max_dist = self.lidar.getMaxRange() // self.block_length
        range_image_vec = np.array(self.lidar.getRangeImage())
        # block_depths = range_image_vec // (self.block_length)
        heading_angle = np.arctan([xy_heading[1] / xy_heading[0]])
        block_d_a = np.column_stack((range_image_vec, np.linspace(heading_angle - (self.field_of_view / 2),
                                                                  heading_angle + (self.field_of_view / 2),
                                                                  self.lidar.getHorizontalResolution())))
        evals = self._map.copy()
        for i in range(self._map.shape[0]):
            for j in range(self._map.shape[1]):
                xy_angle = _xy_angle_calc(robot_loc, (i, j, 0))
                if (xy_angle > heading_angle - (self.field_of_view / 2)) and (
                        xy_angle < xy_heading + (self.field_of_view / 2)) and (
                        np.sqrt((i - robot_loc[0]) ** 2 + (j - robot_loc[1]) ** 2) <= max_dist):
                    # map[i,j,k] = inv_s_m + map[i,j,k] - np.log(prior)
                    pass

    def get(self):
        return self._map

    def get_coordinate(self, coord: tuple[int, int, int]):
        """
        Returns occupancy for specified co-ordinate
        """
        return self._map[coord[0], coord[1], coord[2]]
