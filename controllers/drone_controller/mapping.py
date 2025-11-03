import numpy as np

class Mapping:
    """
    Occupancy grid map
    """

    _map = []
    start_cell: tuple[int, int, int] = (0,0,0)

    def __init__(self, block_length_mm: float, robot_size_blocks: int, lidar):
        self.lidar = lidar
        self.block_length = block_length_mm/1000
        self.field_of_view = lidar.getFov()
        self.robot_size_blocks = robot_size_blocks
        
    def _xy_angle_calc(self, robot_loc_blocks: tuple[int,int,int], spec_loc_blocks: tuple[int,int,int]):
        # calculates angle from x axis for vector between these two points
        lengths = np.array(spec_loc_blocks) - np.array(robot_loc_blocks)
        return np.arctan([lengths[1]/lengths[0]])
        
    def update(self, heading: tuple[int,int], robot_loc: tuple[int,int,int]):
        max_dist = self.lidar.getMaxRange() // self.block_length
        range_image_vec = np.array(self.lidar.getRangeImage())
        # block_depths = range_image_vec // (self.block_length)
        heading_angle = np.arctan([heading[1]/heading[0]])
        block_d_a = np.column_stack((range_image_vec, np.linspace(heading_angle-(self.field_of_view/2), heading_angle+(self.field_of_view/2), self.lidar.getHorizontalResolution())))
        evals = self._map.copy()
        for i in range(self._map.shape[0]):
            for j in range(self._map.shape[1]):
                xy_angle = self._xy_angle_calc(robot_loc, (i,j,0))
                if (xy_angle > heading_angle-(self.field_of_view/2)) and (xy_angle < heading+(self.field_of_view/2)) and (np.sqrt((i-robot_loc[0])**2 + (j-robot_loc[1])**2) <= max_dist):
                    # map[i,j,k] = inv_s_m + map[i,j,k] - np.log(prior)
                    pass
        
    def get(self):
        return self._map
        
    def get_coordinate(self, coord: tuple[int,int,int]):
        """
        Returns occupancy for specified co-ordinate
        """
        return self._map[coord[0], coord[1], coord[2]]