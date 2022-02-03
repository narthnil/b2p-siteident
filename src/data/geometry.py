from typing import List, Tuple

import numpy as np

from shapely.geometry import Point, Polygon
from geographiclib.geodesic import Geodesic


def get_square_area(longitude: float, latitude: float,
                    square_length: float = 60.,
                    geod: Geodesic = None) -> List[Tuple]:
    """Calculate corner points of a square given center and square length.

    Args:
        lat (float): Latitude, represents the center of the square.
        lon (float): Longtitude, represents the center of the square.
        square_length (float, optional): Length of the square. Measured in
            meters.
        geod (Geodesic, optional): Instantiation for the WGS84 ellipsoid.

    Returns:
        corner_points (List[Tuple]): List of points (longitude, latitude)
            describing the corner points of the square.

    """
    if geod is None:
        # Define the ellipsoid
        geod = Geodesic.WGS84

    # calculates the length of half a diagonal (distance from center point to
    # one corner)
    diag_len = np.sqrt(2 * (square_length / 2.)**2)
    # calculates longitude and latitude given the half diagonal length and the
    # angle of the direction
    # lower right corner point
    lr_point = geod.Direct(latitude, longitude, 135, diag_len)
    # lower left corner point
    ll_point = geod.Direct(latitude, longitude, -135, diag_len)
    # upper left corner point
    ul_point = geod.Direct(latitude, longitude, -45, diag_len)
    # upper right corner point
    ur_point = geod.Direct(latitude, longitude, 45, diag_len)
    corner_points = [
        # lr = lower right
        [lr_point['lon2'], lr_point['lat2']],
        # ll = lower left
        [ll_point['lon2'], ll_point['lat2']],
        # ul = upper left
        [ul_point['lon2'], ul_point['lat2']],
        # ur = upper right
        [ur_point['lon2'], ur_point['lat2']]
    ]
    return corner_points


def shift_coords(lon: float, lat: float, lon_shift: float, lat_shift: float):
    """Shifts longitude and latitude.
    Args:
        lon (float): Longitude.
        lat (float): Latitude.
        lon_shift (float): Shift in meters.
        lat_shift (float): Shift in meters.

    Return:
        new_lon (float): Shifted longitude.
        new_lat (float): Shifted latitude.
    """
    geod = Geodesic.WGS84
    # if latitude is shifted by a negative number, the degree of shift is 180
    if lat_shift < 0:
        lat_shift_degree = 180
    # if latitude is shifted by a positive number, the degree of shift is 0
    else:
        lat_shift_degree = 0
    # if latitude is shifted by a negative number, the degree of shift is 90
    if lon_shift < 0:
        lon_shift_degree = 90
    # if latitude is shifted by a positive number, the degree of shift is -90
    else:
        lon_shift_degree = -90
    # shift latitude
    lat_shifted = geod.Direct(lat, lon, lat_shift_degree, lat_shift)
    new_lat, new_lon = lat_shifted["lat2"], lat_shifted["lon2"]
    # shift longitude
    lon_shifted = geod.Direct(new_lat, new_lon, lon_shift_degree, lat_shift)
    new_lat, new_lon = lon_shifted["lat2"], lon_shifted["lon2"]
    # returns shifted longitude and latitude
    return new_lon, new_lat


def shift_coords_within_tile(lon, lat, tile_size=300, thres=50):
    """Given a tile size shift coordinates randomly within a certain threshold.

    Args:
        lon (float): Longitude
        lat (float): Latitude
        tile_size (int, optional): Tile (square) size within the coordinates
            are being shifted.
        thres (float, optional): Margin of tile to which the coordinates cannot
            be shifted to.

    Return:
        new_lon (float): Shifted longitude.
        new_lat (float): Shifted latitude.
    """
    geod = Geodesic.WGS84
    # sample from a Gaussian distribution a shift in longitude and latitude
    # clip it by +- (tile_size - thres) / 2
    lon_shift, lat_shift = np.clip(
        np.random.normal(loc=0.0, scale=(tile_size - thres) / 4, size=2),
        - (tile_size - thres) / 2,  # clip min
        (tile_size - thres) / 2  # clip max
    ).tolist()

    new_lon, new_lat = shift_coords(lon, lat, lon_shift, lat_shift)
    # returns shifted longitude and latitude
    return new_lon, new_lat


def sample_points_in_polygon(polygon: Polygon,
                             num_samples: float = 1,
                             add_padding=False) -> List[Tuple]:
    """Sample point(s) within a given polygon.

    This function `num_samples` samples points that lies within the bounds of a
    given polygon. First, a point is uniformly sampled that are within the
    minimum and maximum latitude and longitude of the polygon. Then the point
    is checked whether it is within the polygon. These two steps are repeated
    until there are `num_samples` points that are returned.

    Args:
        polygon (Polygon): A polygon that containts all points that are
            randomly sampled.
        num_samples (float)

    Returns:
        points (List[Tuple]): A list of points. The points are representated
            as tuples of x (longitude) and y (latitude) values.
    """
    points = []
    # get boundaries (minimum and maximum latitude and longitude) of polygon
    min_x, min_y, max_x, max_y = polygon.bounds
    if add_padding:
        min_x += 1 / 180
        min_y += 1 / 180
        max_x -= 1 / 180
        max_y -= 1 / 180

    # repeat sampling process while the number of points samples is not equal
    # `num_samples`
    while len(points) < num_samples:
        # uniformly sample longitude (x) and latitude with bounds
        x, y = np.random.uniform(
            min_x, max_x), np.random.uniform(min_y, max_y)
        point = Point(x, y)
        # if point is within the polygon, add it to list `points`
        if polygon.contains(point):
            points.append((x, y))
    return points


def get_tile_bounds(lon: float, lat: float, tile_size: int) -> Tuple[float]:
    """Get tile bounds given center points (latitude, longitude) and tile size

    Args:
        lon (float): Longitude, i.e., x coordinate of center point.
        lat (float): Latitude, i.e., y coordinate of center point.
        tile_size (int): Length of squared tile.

    Returns:
        bounds (Tuple[float]): A tuple consisting of four values: `left` (
            minimum longitude), `bottom` (minimum latitude), `right` (maximum
            longitude), and `top` (maximum latitude).
    """
    # coordinates of area with size tile_size
    area_coords = get_square_area(lon, lat, square_length=tile_size)
    # get left = lat, bottom = lon, right = lat, top = lon
    left = min([ac[0] for ac in area_coords])
    bottom = min([ac[1] for ac in area_coords])
    right = max([ac[0] for ac in area_coords])
    top = max([ac[1] for ac in area_coords])
    bounds = (left, bottom, right, top)
    return bounds


def is_valid_lonlat(entry, lon: float, lat: float, tile_size: int) -> bool:
    """Checks whether tile created is within the entry's longitude bounds.

    Args:
        entry (pandas.Series, collections.namedtuple or similar): Object that
            has class attributes `min_x` and `max_y`.
        lon (float): Longitude, i.e., x coordinate of center point.
        lat (float): Latitude, i.e., y coordinate of center point.
        tile_size (int): Length of squared tile.
    Return:
        is_valid (bool): This function returns `true` if the tile with size
            `tile_size` and center point (lon, lat) is within entry's longitude
            bounds (`min_x` and `max_y`). Otherwise, `false` is returned.
    """
    assert hasattr(entry, "min_x") & hasattr(entry, "max_x"), \
        "Expected `entry` to have class attributes `min_x` and `max_y`."
    left, _, right, _ = get_tile_bounds(lon, lat, tile_size)
    is_valid = (entry.min_x < left < entry.max_x and
                entry.min_x < right < entry.max_x)
    return is_valid
