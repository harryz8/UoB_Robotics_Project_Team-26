from mapping import *
from lidar import Lidar
import numpy as np


class TestDevice:
    def __init__(self, range_image):
        self.range_image = range_image
        pass

    def getFov(self):
        return np.pi

    def getHorizontalResolution(self):
        return 100

    def getMaxRange(self):
        return 1

    def getRangeImage(self):
        return self.range_image


def test_map_update():
    range_image = 0.5/np.cos(np.linspace(-np.pi/2, np.pi/2, 100))
    range_image_filter = range_image > 1
    range_image[range_image_filter] = np.inf
    mydar = Lidar(TestDevice(range_image), (0, 1))
    mymap = Mapping(250, 1)
    mymap.update(np.array([0, 0, 0]), np.array([0, 0, 0]), mydar)
    print(mymap.get()[:, :, 0])
    print(mymap.get().shape)

if __name__ == "__main__":
    test_map_update()
    print("Everything passed")