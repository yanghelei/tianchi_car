from shapely import affinity
from shapely.geometry import Polygon
from math import degrees
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