import numpy as np
import math


def _xy_angle_calc(robot_loc_blocks: tuple[int, int, int], spec_loc_blocks: tuple[int, int, int]) -> float:
    # calculates angle from x-axis for vector between these two points
    lengths = np.array(spec_loc_blocks) - np.array(robot_loc_blocks)
    return np.arctan2([lengths[1]],[lengths[0]])


class Mapping:
    """
    Occupancy grid map
    """

    _map = np.ones((1, 1, 1), dtype='f')
    start_cell = np.array([0, 0, 0])

    def __init__(self, block_length_mm: float, robot_size_blocks: int, lidar):
        self.lidar = lidar
        self.block_length = block_length_mm / 1000
        self.field_of_view = lidar.getFov()
        self.robot_size_blocks = robot_size_blocks

    def _extend_to(self, coords: tuple[int, int, int]) -> np.ndarray:
        prev_start = self.start_cell.copy()
        map_shape = (self._map.shape - np.ones(3)).astype('i')
        self.start_cell = self.start_cell + (map_shape - np.maximum(map_shape, np.array(coords))) - (np.zeros(3) + np.minimum(np.zeros(3), np.array(coords)))
        # print(f"({abs(min(0, coords[0]))}, {max(0, coords[0]-map_shape[0])}), ({abs(min(0, coords[1]))}, {max(0, coords[1]-map_shape[1])}), ({abs(min(0, coords[2]))}, {max(0, coords[2]-map_shape[2])})")
        self._map = np.pad(self._map, pad_width=(
        (abs(min(0, coords[0])), max(0, coords[0]-map_shape[0])),
        (abs(min(0, coords[1])), max(0, coords[1]-map_shape[1])),
        (abs(min(0, coords[2])), max(0, coords[2]-map_shape[2]))), 
        mode = 'constant', constant_values = 1)
        return self.start_cell - prev_start

    def update(self, heading_angle: float, robot_loc: np.ndarray):
        """
        Update map after new lidar reading.
        
        Args:
            heading_angle (float): The angle between the front of the drone and the x-axis, in radians
            robot_loc (np.ndarray): A 3 valued vector of type np.array which holds the x,y,z positions of the drone from its starting point in meters
            
        Returns:
            None
        """
        learning_rate: int = 1
        # extend map
        # diff_loc = robot_loc - self.start_cell
        new_blocks_max_dist: int = math.ceil(self.lidar.getMaxRange() / self.block_length)
        new_blocks: list[tuple[int, int, int]] = [(x,y,z) for x in range(robot_loc[0]-new_blocks_max_dist, robot_loc[0]+new_blocks_max_dist+1) for y in range(robot_loc[1]-new_blocks_max_dist, robot_loc[1]+new_blocks_max_dist+1) for z in range(robot_loc[2]-new_blocks_max_dist, robot_loc[2]+new_blocks_max_dist+1)]
        change_vec = np.zeros(3)
        for new_block in new_blocks:
            if not((new_block[0]+change_vec[0]<self._map.shape[0]) and (new_block[1]+change_vec[1]<self._map.shape[1]) and (new_block[2]+change_vec[2]<self._map.shape[2]) and (new_block[0]+change_vec[0]>=0) and (new_block[1]+change_vec[1]>=0) and (new_block[2]+change_vec[2]>=0)):
                change_vec = change_vec + self._extend_to((int(new_block[0]+change_vec[0]),int(new_block[1]+change_vec[1]),int(new_block[2]+change_vec[2])))
                # robot_loc = robot_loc + diff_loc
        robot_loc = (robot_loc + change_vec).astype("i")
        # get lidar values
        max_dist = self.lidar.getMaxRange() // self.block_length
        range_image_vec = np.array(self.lidar.getRangeImage())
        # block_depths = range_image_vec // (self.block_length)
        readings = np.column_stack((range_image_vec, np.linspace(heading_angle - (self.field_of_view / 2),
                                                                  heading_angle + (self.field_of_view / 2),
                                                                  self.lidar.getHorizontalResolution())))
        # evals = self._map.copy()
        # update map
        for i in range(self._map.shape[0]):
            for j in range(self._map.shape[1]):
                xy_angle = _xy_angle_calc(robot_loc, (i, j, 0))[0]%(2*math.pi)
                # print(f"x:{i}, y:{j}, angle:{xy_angle}, ha:{((heading_angle%(2*math.pi)) - (self.field_of_view / 2))}, ha2:{((heading_angle%(2*math.pi)) + (self.field_of_view / 2))}")
                if (xy_angle > ((heading_angle%(2*math.pi)) - (self.field_of_view / 2))) and (
                        xy_angle < ((heading_angle%(2*math.pi)) + (self.field_of_view / 2))) and (
                        np.sqrt((i - robot_loc[0]) ** 2 + (j - robot_loc[1]) ** 2) <= max_dist):
                    # print(f"Len: {len(readings)}")
                    for reading in readings:
                        reading_x: float = ((reading[0]*np.cos(reading[1]))/self.block_length) + robot_loc[0]
                        reading_y: float = ((reading[0]*np.sin(reading[1]))/self.block_length) + robot_loc[1]
                        if (i - reading_x < 0) and (i - reading_x > -1) and (j - reading_y < 0) and (j - reading_y > -1):
                            ism = learning_rate/2
                        elif (i - reading_x < -1) and (j - reading_y < -1):
                            ism = - learning_rate/2
                        else:
                            ism = 0
                        if ism != 0:
                            print(f"x:{i}, y:{j}, angle:{xy_angle}, ha:{((heading_angle%(2*math.pi)) - (self.field_of_view / 2))}, ha2:{((heading_angle%(2*math.pi)) + (self.field_of_view / 2))}")
                            print(f"Ism: {ism}")
                            print(self._map)
                        self._map[i,j,robot_loc[2]] = ism + self._map[i,j,robot_loc[2]] - 0 # let 0 = prior

    def get(self):
        return self._map

    def get_coordinate(self, coord: tuple[int, int, int]) -> np.ndarray:
        """
        Returns occupancy for specified co-ordinate
        """
        return self._map[coord[0], coord[1], coord[2]]
