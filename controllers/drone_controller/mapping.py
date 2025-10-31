import numpy as np

class Mapping:
    """
    Occupancy grid map
    """

    map = []
    start_cell : tuple[int, int, int] = (0,0,0)

    def __init__(self, block_length_mm : float, robot_size_blocks : int, lidar):
        self.lidar = lidar
        self.block_length = block_length_mm
        self.field_of_view = lidar.getFov()
        self.max_range = lidar.getMaxRange()
        
    def update(self):
        range_image_vec = np.array(self.lidar.getRangeImage())
        block_depths = range_image_vec // (self.block_length/1000)
        
        
    def get(self):
        return np.array(map)
        
    def get_coordinate(self, x, y, z):
        """
        Returns occupancy for specified co-ordinate
        """
        return map[x,y,z]