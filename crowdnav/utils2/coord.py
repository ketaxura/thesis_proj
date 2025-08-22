# crowdnav/utils2/coord.py
import numpy as np

def grid_to_world(idx, res, shape):
    half_x = res * shape[1] / 2.0
    half_y = res * shape[0] / 2.0
    r, c = idx
    x = (c + 0.5) * res - half_x
    y = (r + 0.5) * res - half_y
    return np.array([x, y])

def world_to_grid(xy, res, shape):
    half_x = res * shape[1] / 2.0
    half_y = res * shape[0] / 2.0
    c = int((xy[0] + half_x)/res)
    r = int((xy[1] + half_y)/res)
    r = np.clip(r, 0, shape[0]-1); c = np.clip(c, 0, shape[1]-1)
    return int(r), int(c)

def wrap_angle(a):
    import numpy as np
    return (a + np.pi) % (2*np.pi) - np.pi
