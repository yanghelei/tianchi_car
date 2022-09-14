import copy
import numpy as np

import torch
import torch.nn as nn


def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def check(input):
    output = torch.from_numpy(input) if type(input) == np.ndarray else input
    return output


def orthogonal_init_(m, gain=np.sqrt(2)):
    return init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), gain=gain)