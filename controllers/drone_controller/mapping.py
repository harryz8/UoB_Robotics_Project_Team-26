import numpy as np

class Mapping:

    map = []
    start_cell : tuple[int, int, int] = (0,0,0)

    def __init__(self, block_length_mm : int):
        self.block_length = block_length_mm
        
    def getMap(self):
        return np.array(map)