import numpy as np

class Mapping:
    """
    Occupancy grid map
    """

    map = []
    start_cell : tuple[int, int, int] = (0,0,0)

    def __init__(self, block_length_mm : float, robot_size_blocks : int, lidar):
        self.lidar = lidar
        self.block_length = block_length_mm/1000
        self.field_of_view = lidar.getFov()
        
    def _xy_angle_calc(robot_loc_blocks : tuple[int,int,int], spec_loc_blocks : tuple[int,int,int]):
        # calculates angle from x axis for vector between these two points
        lengths = np.array(spec_loc_blocks) - np.array(robot_loc_blocks)
        return np.arctan([lengths[1]/lengths[0]])
        
    def update(self, heading : tuple[int,int], robot_loc : tuple[int,int,int]):
        max_dist = self.lidar.getMaxRange() // self.block_length
        range_image_vec = np.array(self.lidar.getRangeImage())
        # block_depths = range_image_vec // (self.block_length)
        block_d_a = np.column_stack((range_image_vec, np.linspace(heading-(self.field_of_view/2), heading+(self.field_of_view/2), self.lidar.getHorizontalResolution())))
        evals = map.copy()
        for i in range(map.shape[0]):
            for j in range(map.shape[1]):
                xy_angle = _xy_angle_calc(robot_loc, (x,y,0))
                if (xy_angle > heading-(self.field_of_view/2)) and (xy_angle < heading+(self.field_of_view/2)) and (numpy.sqrt((i-robot_loc[0])**2 + (j-robot_loc[1])**2) <= max_dist):
                    # map[i,j,k] = inv_s_m + map[i,j,k] - np.log(prior)
        
    def get(self):
        return np.array(map)
        
    def get_coordinate(self, x, y, z):
        """
        Returns occupancy for specified co-ordinate
        """
        return map[x,y,z]