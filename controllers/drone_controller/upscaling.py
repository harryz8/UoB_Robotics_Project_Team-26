import numpy as np

def linear_upscaling(image_map: np.ndarray, factor: int):
    new_image_map_shape = np.array(image_map.shape) * factor
    new_image_map = np.zeros(new_image_map_shape)
    # print(image_map)
    for i in range (image_map.shape[0]):
        for j in range (image_map.shape[1]):
            new_image_map[factor*i, factor*j] = 0.25*0.25*image_map[i-1, j-1] + 0.25*0.75*image_map[i-1, j] + 0.75*0.25*image_map[i, j-1] + 0.75*0.75*image_map[i, j]
            new_image_map[factor*i+1, factor*j] = 0.25*0.25*image_map[i, j-1] + 0.25*0.75*image_map[i, j] + 0.75*0.25*image_map[i-1, j-1] + 0.75*0.75*image_map[i-1, j]
            new_image_map[factor*i, factor*j+1] = 0.25*0.25*image_map[i-1, j] + 0.25*0.75*image_map[i-1, j-1] + 0.75*0.25*image_map[i, j] + 0.75*0.75*image_map[i, j-1]
            new_image_map[factor*i+1, factor*j+1] = 0.25*0.25*image_map[i, j] + 0.25*0.75*image_map[i, j-1] + 0.75*0.25*image_map[i-1, j] + 0.75*0.75*image_map[i-1, j-1]

    return new_image_map

if __name__ == "__main__":
    mymap = np.array([[1, 0, 1],
                    [1, 0, 1],
                    [1, 0, 1]])
    print(linear_upscaling(mymap, 2))