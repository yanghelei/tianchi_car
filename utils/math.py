import numpy as np


def compute_distance(pos_0, pos_1):
    return np.sqrt((pos_0[0] - pos_1[0]) ** 2 + (pos_0[1] - pos_1[1]) ** 2)

