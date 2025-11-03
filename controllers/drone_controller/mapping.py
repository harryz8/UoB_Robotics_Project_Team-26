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
        
    def _valid_map_index(index: Tuple[int,int,int]):
        index_arr = np.array(index)
        return (index_arr >= 0).all() and (index_arr < self.map.shape).all()
        
    def update(self, ism, prior, loc):
        current_cell = (loc[0]/self.block_length, loc[1]/self.block_length, loc[2]/self.block_length)
        range_image_vec = np.array(self.lidar.getRangeImage())
        block_depths = range_image_vec // (self.block_length/1000)
        self.map = np.log(ism / (1-ism)) + np.log((1-prior)/prior) + self.map
        return np.array(self.map)
        
    def get(self):
        return np.array(map)
        
    def get_coordinate(self, x, y, z):
        """
        Returns occupancy for specified co-ordinate
        """
        if not self._valid_map_index((x,y,z)):
            raise IndexError("Index out of range for this map")
        return map[x,y,z]