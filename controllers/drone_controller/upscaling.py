import numpy as np

def linear_upscaling(image_map: np.ndarray, factor: int):
    new_image_map_shape = np.array(image_map.shape) * factor
    new_image_map = np.zeros(new_image_map_shape)
    # print(image_map)
    for i in range (image_map.shape[0]):
        p = (i + 1) % image_map.shape[1]
        for j in range (image_map.shape[1]):
            q = (j+1) % image_map.shape[1]
            new_image_map[factor*i+1, factor*j+1] = 0.25*0.25*image_map[i, j] + 0.25*0.75*image_map[i, q] + 0.75*0.25*image_map[p, j] + 0.75*0.75*image_map[p, q]
            print(f"map[i,j]: {image_map[i, j]}")
            print(f"map[i-1,j]: {image_map[i-1, j]}")
            print(f"map[i-1,j-1]: {image_map[i-1, j-1]}")
            print(f"map[i,j-1]: {image_map[i, j-1]}")
            print(new_image_map[factor*i, factor*j])
            new_image_map[factor*i, factor*j+1] = 0.25*0.25*image_map[p, j] + 0.25*0.75*image_map[p, q] + 0.75*0.25*image_map[i, j] + 0.75*0.75*image_map[i, q]
            print(new_image_map[factor * i+1, factor * j])
            new_image_map[factor*i+1, factor*j] = 0.25*0.25*image_map[i, q] + 0.25*0.75*image_map[i, j] + 0.75*0.25*image_map[p, q] + 0.75*0.75*image_map[p, j]
            print(new_image_map[factor * i, factor * j+1])
            new_image_map[factor*i, factor*j] = 0.25*0.25*image_map[p, q] + 0.25*0.75*image_map[p, j] + 0.75*0.25*image_map[i, q] + 0.75*0.75*image_map[i, j]
            print(new_image_map[factor * i, factor * j])

    return np.roll(new_image_map, 1, axis=1)

if __name__ == "__main__":
    mymap = np.array([[1, -1, 1],
                      [1, -1, 1],
                      [1, -1, 1]])
    print(linear_upscaling(mymap, 2))