import torch.nn as nn
from utils.util import init, get_clones
import numpy as np 

"""MLP modules."""

def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        init_method = nn.init.orthogonal_
        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain=np.sqrt(2))
        layers += [init_(nn.Linear(sizes[j], sizes[j+1]), act())]
    return nn.Sequential(*layers)

class MLPLayer(nn.Module):
    def __init__(self, input_dim, hidden_size, layer_N, use_orthogonal, use_ReLU):
        super(MLPLayer, self).__init__()
        self._layer_N = layer_N
        
        # 激活函数
        active_func = [nn.Tanh(), nn.ReLU()][use_ReLU]
        # 网络初始化方法
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][use_orthogonal]
        gain = nn.init.calculate_gain(['tanh', 'relu'][use_ReLU])

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain=gain)

        self.fc1 = nn.Sequential(
            init_(nn.Linear(input_dim, hidden_size)), active_func)
        fc_h = nn.Sequential(init_(
            nn.Linear(hidden_size, hidden_size)), active_func)
        self.fc2 = get_clones(fc_h, self._layer_N)

    def forward(self, x):
        x = self.fc1(x)
        for i in range(self._layer_N):
            x = self.fc2[i](x)
        return x

class MLPBase(nn.Module):
    def __init__(self, obs_dim, hidden_size, layer_N, use_feature_normalization=False):
        super(MLPBase, self).__init__()

        self._use_orthogonal = True
        self._use_ReLU = True
        self._layer_N = layer_N
        self.hidden_size =hidden_size
        self._use_feature_normalization = use_feature_normalization

        if self._use_feature_normalization:
            self.feature_norm = nn.LayerNorm(obs_dim)

        self.mlp = MLPLayer(obs_dim, self.hidden_size,
                              self._layer_N, self._use_orthogonal, self._use_ReLU)

    def forward(self, x):
        if self._use_feature_normalization:
            x = self.feature_norm(x)

        x = self.mlp(x)

        return x