from math import pi, sqrt, degrees
from shapely import affinity
from shapely.geometry import Polygon, LineString


def compute_distance(pos_0, pos_1):
    return sqrt((pos_0[0] - pos_1[0]) ** 2 + (pos_0[1] - pos_1[1]) ** 2)


def wrap(angle):
    while angle >= pi:
        angle -= 2 * pi
    while angle < -pi:
        angle += 2 * pi
    return angle


def wrap_star(angle):
    angle = pi - angle
    return wrap(angle)


def get_polygon(center, length, width, theta):
    x, y = center
    polygon_points = (
        (x + length / 2, y + width / 2),
        (x + length / 2, y - width / 2),
        (x - length / 2, y - width / 2),
        (x - length / 2, y + width / 2),
    )
    polygon = Polygon(polygon_points)
    return affinity.rotate(polygon, angle=round(degrees(theta), 2))
