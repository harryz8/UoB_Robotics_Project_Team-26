import numpy as np

class MapFactory:
    def __init__(self, prior_map: np.ndarray):
        self.__map = prior_map
        # prepare location
