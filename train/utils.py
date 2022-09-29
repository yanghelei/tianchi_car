import numpy as np
import itertools
from shapely import affinity
from shapely.geometry import Polygon, LineString
import math


def wrap(angle):
    '''
    将智能体的角度限制在-pi-pi
    '''
    while angle >= np.pi:
        angle -= 2 * np.pi
    while angle < - np.pi:
        angle += 2 * np.pi
    return angle


def cal_vector_angle(vec1, vec2, eta=1e-6):
    '''
    计算两个向量之间的夹角0-pi
    加上一个非常小的正数防止数值不稳定
    '''
    cos_value = vec1.dot(vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2) + eta)
    if 1.0 < cos_value < 1.0 + eta:
        cos_value = 1.0
    elif -1.0 - eta < cos_value < -1.0:
        cos_value = -1.0
    return np.arccos(cos_value)


def change_heading_angle_to_vec(heading_angle):
    unit = 1
    return np.array([unit * np.cos(heading_angle), unit * np.sin(heading_angle)])


def cal_relative_rotation_angle(pos1: np.array, pos2: np.array, heading_angle):
    """
    pos1: 当前智能体位置
    pos2: 目标智能体位置
    heading_angle: 当前智能体朝向角
    return: 当前智能体到目标智能体位置应作出的转角-pi-pi
    """
    heading_vec = change_heading_angle_to_vec(heading_angle)
    eta = 1e-6

    pos_vec = pos2 - pos1
    angle = cal_vector_angle(heading_vec, pos_vec)
    # 计算向量的叉积
    sin_value = np.cross(heading_vec, pos_vec) / (np.linalg.norm(heading_vec) * np.linalg.norm(pos_vec) + eta)
    if 1 < sin_value < 1 + eta:
        sin_value = 1.0
    elif -1 - eta < sin_value < -1:
        sin_value = -1.0
    rho = np.arcsin(sin_value)
    return angle if rho > 0 else wrap(2 * np.pi - angle)


def compute_distance(pos_0, pos_1):
    return np.sqrt((pos_0[0] - pos_1[0]) ** 2 + (pos_0[1] - pos_1[1]) ** 2)


class ActionProjection:
    def __init__(self, low, high, action_per_dim):
        mesh = [np.linspace(lo, hi, a) for lo, hi, a in zip(low, high, action_per_dim)]
        self.action_library = list(itertools.product(*mesh))

    def get_action(self, act):
        assert len(act.shape) <= 2, f"Unknown action format with shape {act.shape}."
        if len(act.shape) == 1:
            return np.array([self.action_library[a] for a in act])
        return np.array([[self.action_library[a] for a in a_] for a_ in act])


def get_polygon(center, length, width, theta):
    x, y = center
    polygon_points = (
        (x + length / 2, y + width / 2),
        (x + length / 2, y - width / 2),
        (x - length / 2, y - width / 2),
        (x - length / 2, y + width / 2),
    )
    polygon = Polygon(polygon_points)
    return affinity.rotate(polygon, angle=round(math.degrees(theta), 2))


a = ActionProjection([0, 0], [1, 1], [5, 5])
a = np.array(list(map(list, a.action_library)))
print()
