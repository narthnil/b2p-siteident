import json
import itertools

import numpy as np
import geopandas as gpd
import rasterio

import torch

from rasterio import windows
from typing import Dict, List, Tuple, Iterator
from torchvision import transforms

from geographiclib.geodesic import Geodesic
from shapely.geometry import Point, Polygon

from torch.utils.data import Dataset, Sampler, DataLoader


# key is the tile size (300, 600 or 1200m) and values are tuples representing
# the width and height of the corresponding tile / image
OUTPUT_SIZE = {300: (12, 12), 600: (24, 24), 1200: (48, 48)}

# fixed order of which data modality is used and in which order
# this serves as default order
DATA_ORDER = [
    "population",
    "osm_img",
    "elevation",
    "slope",
    "roads",
    "waterways",
    "admin_bounds_qgis"]

# this dictionary contains for each country the data modality with its data
# path and the target channels that can be read with rasterio
METADATA = {
    "Rwanda": {
        "population": {
            "fp": ("./data/population/"
                   "Rwanda_population_data_2020_UNadj_constrained.tif"),
            "raster_channels": [1]
        },
        "osm_img": {
            "fp": "./data/osm/imgs/rwanda_osm_nolab_1-50000_4326.tiff",
            "raster_channels": [1, 2, 3]
        },
        "elevation": {
            "fp": "./data/slope_elevation/elevation_rwanda.tif",
            "raster_channels": [2]
        },
        "slope": {
            "fp": "./data/slope_elevation/slope_rwanda.tif",
            "raster_channels": [2]
        },
        "roads": {
            "fp": "./data/osm/roads/rwanda-osm-roads.tif",
            "raster_channels": [1]
        },
        "waterways": {
            "fp": "./data/osm/waterways/rwanda-osm-waterways.tif",
            "raster_channels": [1]
        },
        "admin_bounds_qgis": {
            "fp": ("./data/admin_boundaries/"
                   "Rwanda_Village_AdminBoundary_1_3600.tiff"),
            "raster_channels": [1]
        },
        "admin_bounds_gadm": {
            "fp": "./data/admin_boundaries/GADM_rwanda.tif",
            "raster_channels": [1]
        },
        "country_bounds": {
            "fp": "./data/country_masks/rwanda.shp",
            "raster_channels": [1]
        }
    },
    "Uganda": {
        "population": {
            "fp": ("./data/population/"
                   "Uganda_population_data_2020_UNadj_constrained.tif"),
            "raster_channels": [1]
        },
        "osm_img": {
            "fp": "./data/osm/imgs/uganda_osm_nolab_1-50000_4326.tiff",
            "raster_channels": [1, 2, 3]
        },
        "elevation": {
            "fp": "./data/slope_elevation/elevation_uganda.tif",
            "raster_channels": [2]
        },
        "slope": {
            "fp": "./data/slope_elevation/slope_uganda.tif",
            "raster_channels": [2]
        },
        "roads": {
            "fp": "./data/osm/roads/uganda-osm-roads.tif",
            "raster_channels": [1]
        },
        "waterways": {
            "fp": "./data/osm/waterways/uganda-osm-waterways.tif",
            "raster_channels": [1]
        },
        "admin_bounds_qgis": {
            "fp": ("./data/admin_boundaries/"
                   "Uganda_Parish_AdminBoundary_1_3600.tiff"),
            "raster_channels": [1]
        },
        "admin_bounds_gadm": {
            "fp": "./data/admin_boundaries/GADM_uganda.tif",
            "raster_channels": [1]
        },
        "country_bounds": {
            "fp": "./data/country_masks/uganda.shp",
            "raster_channels": [1]
        }
    },
}

TRAIN_DATA = {
    "v1": {
        300: "./data/ground_truth/train_300_v1_ssl.geojson",
        600: "./data/ground_truth/train_600_v1_ssl.geojson",
        1200: "./data/ground_truth/train_1200_v1_ssl.geojson"
    },
    "v2": {
        300: "./data/ground_truth/train_300_v2_ssl.geojson",
        600: "./data/ground_truth/train_600_v2_ssl.geojson",
        1200: "./data/ground_truth/train_1200_v2_ssl.geojson"
    }

}

STATS_FP = "./data/ground_truth/stats.json"
THRES = 50


def get_num_channels(data_modalities: List[str]) -> int:
    """Returns total number of channels given data modalities
    """
    num_channels = 0
    for name in data_modalities:
        if name not in METADATA["Rwanda"]:
            raise Exception("Data modality {} not known.".format(name))
        num_channels += len(METADATA["Rwanda"][name]["raster_channels"])
    return num_channels


class UnNormalize(object):
    """Undo normalizing a tensor with a mean and standard deviation
    Normalize a Tensor with mean and standard deviation.

    Given mean: (mean[1],...,mean[n]) and std: (std[1],..,std[n]) for n
    channels, this transform will undo the normalization of each channel of the
    input torch.*Tensor i.e.,
        output[channel] = input[channel] * std[channel] + mean[channel])
    """

    def __init__(self, mean: List[float], std: List[float]) -> None:
        self.mean = mean
        self.std = std

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor


class Normalize(object):
    """Normalize a Tensor with mean and standard deviation.

    Given mean: (mean[1],...,mean[n]) and std: (std[1],..,std[n]) for n
    channels, this transform will normalize each channel of the input
    torch.*Tensor i.e.,
        output[channel] = (input[channel] - mean[channel]) / std[channel]
    """

    def __init__(self, mean: List[float], std: List[float]) -> None:
        self.mean = mean
        self.std = std

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        i = 0
        for t, m, s in zip(tensor, self.mean, self.std):
            t.sub_(m).div_(s)
            i += 1
        return tensor


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


def shift_coords(lon, lat, tile_size=300, thres=THRES):
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
    lat_shift, lon_shift = np.clip(
        np.random.normal(loc=0.0, scale=(tile_size - thres) / 4, size=2),
        - (tile_size - thres) / 2,  # clip min
        (tile_size - thres) / 2  # clip max
    ).tolist()
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


def augment_pop_data(img: np.ndarray, index_pop: int, stats: Dict,
                     loc: float = 0, scale: float = 3) -> np.ndarray:
    """Augments population data.

    The population data is augmented by random integer values that are Gaussian
    distributed. Finally we clip the values by the minimum and maximum
    population values according to the statistics.

    Args:
        img (np.ndarray): An array with shape height x width x channels. The
            index_pop-th channel contains the population data.
        index_pop (int): Index of the population channel.
        stats (Dict): Dictionary containing the minimum (`min`) and maximum
            (`max`) values for clipping the population values.
        loc (float, optional): Mean of the Gaussian distribution used to sample
            random integers to augment the population data. Default: 0.
        scale (float, optional): Standard deviation of the Gaussian
            distribution used to sample random integers to augment the
            population data. Default: 3.

    Returns:
        img (np.ndarray): An array with shape height x width x channels. The
            population channel was augmented.
    """
    assert img.ndim == 3, "Expected `img` to have 3 dimensions."
    assert img.shape[2] >= index_pop, "Index cannot be accessed"
    assert "min" in stats and "max" in stats, \
        "stats does not have min and max keys."

    height, width, _ = img.shape
    # augments with rounded samples of Gaussian distribution N(loc, scale)
    img[:, :, index_pop:index_pop + 1] += np.random.normal(
        loc=loc, scale=scale, size=(height, width, 1)).round()
    # clip all values by stats min and max values for population
    img[:, :, index_pop:index_pop + 1] = np.clip(
        img[:, :, index_pop:index_pop + 1], stats["min"][0], stats["max"][0])
    return img


def augment_terrain_data(img: np.ndarray, index_terrain: int, loc: float = 0,
                         scale: float = 2) -> np.ndarray:
    """Augments terrain data.

    We augment each *unique* terrain (elevation / slope) value by a randomly
    Gaussian distributed value. The same terrain value gets the same random
    number. We sort the unique terrain values and we sort the random numbers.
    Thus, smaller terrain values gets a smaller change than larger terrain
    values. In this way we make sure that the relative order of terrain values
    will not be changed through the augmentation.

    Args:
        img (np.ndarray): An array with shape height x width x channels. The
            index_terrain-th channel contains the terrain data.
        index_terrain (int): Index of the terrain channel.
        loc (float, optional): Mean of the Gaussian distribution used to sample
            random integers to augment the terrain data. Default: 0.
        scale (float, optional): Standard deviation of the Gaussian
            distribution used to sample random integers to augment the
            terrain data. Default: 2.

    Returns:
        img (np.ndarray): An array with shape height x width x channels. The
            terrain channel was augmented.
    """
    assert img.ndim == 3, "Expected `img` to have 3 dimensions."
    assert img.shape[2] >= index_terrain, "Index cannot be accessed"

    img_terrain = img[:, :, index_terrain:index_terrain + 1]
    img_terrain_new = np.array(img[:, :, index_terrain:index_terrain + 1])
    # sort the unique terrain data
    unique_elems = sorted(np.unique(img_terrain).tolist())
    # sort the random samples from a Gaussian distribution
    # N(loc, scale)
    rnd = sorted(np.random.normal(
        loc=loc, scale=scale, size=len(unique_elems)).round().tolist())
    # add random samples to the terrain data
    for i, elem in enumerate(unique_elems):
        img_terrain_new[np.where(img_terrain == elem)] += rnd[i]
    img[:, :, index_terrain:index_terrain + 1] = img_terrain_new
    return img


def augment_osm_img(img: np.ndarray, index_img: int, loc: float = 0,
                    scale: float = 3) -> np.ndarray:
    """Augments OSM image data.

    The OSM image data is augmented by random integer values that are Gaussian
    distributed. Finally we clip the values by the minimum and maximum
    image pixel values (0 and 255).

    Args:
        img (np.ndarray): An array with shape height x width x channels. The
            index_img-th channel contains the OSM image data.
        index_img (int): Index of the OSM image channel.
        loc (float, optional): Mean of the Gaussian distribution used to sample
            random integers to augment the OSM image data. Default: 0.
        scale (float, optional): Standard deviation of the Gaussian
            distribution used to sample random integers to augment the OSM
            image data. Default: 2.

    Returns:
        img (np.ndarray): An array with shape height x width x channels. The
            OSM image channel was augmented.
    """
    assert img.ndim == 3, "Expected `img` to have 3 dimensions."
    assert img.shape[2] >= index_img, "Index cannot be accessed"
    height, width, _ = img.shape
    # augment with rounded Gaussian distributed samples N(loc, scale)
    img[:, :, index_img:index_img + 3] += np.random.normal(
        loc=loc, scale=scale, size=(height, width, 3)).round()
    # clip all values by min=0 and max=255
    img[:, :, index_img:index_img + 3] = np.clip(
        img[:, :, index_img:index_img + 3], 0, 255)
    return img


def augment_binary_map(img: np.ndarray, index_map: int,
                       p_reject: float = 0.5) -> np.ndarray:
    """Augments binary map data.

    A binary map has only two values, 0 and 1. We augment these maps by setting
    values 1 to 0 with a probability given by `p_reject`.

    Args:
        img (np.ndarray): An array with shape height x width x channels. The
            index_map-th channel contains the binary data.
        index_img (int): Index of the binary channel.
        p_reject (float, optional): Probability that the pixel 1 is not flipped
            to 0. Default: 0.5.

    Returns:
        img (np.ndarray): An array with shape height x width x channels. The
            binary map channel was augmented.
    """
    assert img.ndim == 3, "Expected `img` to have 3 dimensions."
    assert img.shape[2] >= index_map, "Index cannot be accessed"
    assert 0 <= p_reject <= 1, \
        "Expected probability `p_reject` to be within [0, 1]."

    height, width, _ = img.shape
    # randomly sample a number between 0 and 1 for each pixel
    p = np.random.rand(height, width, 1)
    img_binary = img[:, :, index_map:index_map + 1]
    # if p > p_reject and the pixel is 1, then set it to 0
    img_binary[np.where(np.logical_and(p > p_reject, img_binary > 0))] = 0
    img[:, :, index_map:index_map + 1] = img_binary
    return img


class BridgeDataset(Dataset):
    """Dataset module to load all TIF-based data from file and extract tiles.
    """

    def __init__(self, data: Dict = TRAIN_DATA,
                 data_order: List[str] = DATA_ORDER, data_version: str = "v1",
                 raster_data: Dict = METADATA, stats_fp: str = STATS_FP,
                 tile_size: int = 300, transform: bool = True,
                 use_augment: bool = True,
                 use_rnd_center_point: bool = True) -> None:
        """
        Args:
            data (Dict, optional): Dictionary contains the train data file path
                for the `data_version` and `tile_size`. Default: TRAIN_DATA.
            data_order (str, optional): The order of the data modalities to be
                loaded and read. Default: DATA_ORDER.
            data_version (str, optional): The version of the data, can be
                either `v1` or `v2`. Default: v1.
            raster_data (Dict, optional): This dictionary contains for each
                country the data modality with its data path and the target
                channels that can be read with rasterio. Default: METADATA.
            stats_fp (str, optional): The file path of the data stats. Default:
                STATS_FP.
            tile_size (int, optional): Tile (square) size, can be either 300,
                600, or 1200 metres. Default: 300.
            transform (bool, optional): Whether to normalize the data or not.
                Default: True.
            use_augment (bool, optional): Whether to use augmentation on the
                data or not. Default: True.
            use_rnd_center_point (bool, optional): Whether to use random tile
                center points or not. Default: True.
        """
        assert tile_size in [300, 600, 1200], "Tile size not known."
        assert data_version in ["v1", "v2"], "Data version not known."
        assert data_version in data and tile_size in data[data_version], \
            "Expected for data[data_version][tile_size] to exist."

        self.data_order = data_order
        self.tile_size = tile_size
        self.use_rnd_center_point = use_rnd_center_point
        self.train_metadata = raster_data
        self.transform = transform
        self.use_augment = use_augment

        # load training data
        with open(data[data_version][tile_size]) as f:
            self.train_gdf = gpd.read_file(f)

        # open each dataset for reading and save to self.data_rasters
        # skip `country_bounds`
        self.data_rasters = {}
        for country, data_modalities in raster_data.items():
            if country not in self.data_rasters:
                self.data_rasters[country] = {}
            for data_type, data in data_modalities.items():
                if data_type == "country_bounds":
                    continue
                self.data_rasters[country][data_type] = rasterio.open(
                    data["fp"])

        # load statistics
        with open(stats_fp) as f:
            self.stats = json.load(f)

        # normalization function
        self.transform_func = transforms.Compose([
            Normalize(
                list(itertools.chain(
                    *[self.stats[name]["mean"] for name in self.data_order])),
                list(itertools.chain(
                    *[self.stats[name]["std"] for name in self.data_order])))

        ])
        # undo normalization function
        self.invert_transform_func = transforms.Compose([
            UnNormalize(
                list(itertools.chain(
                    *[self.stats[name]["mean"] for name in self.data_order])),
                list(itertools.chain(
                    *[self.stats[name]["std"] for name in self.data_order])))

        ])

    def convert_tensor_2_numpy(self, imgs: torch.Tensor) -> np.ndarray:
        """Given a dataset item, unnormalize if necessary and return numpy.

        Args:
            img (torch.Tensor): An array with shape channels x height x width.

        Returns:
            img (np.ndarray): An array with shape height x width x channels.
        """
        # invert normalization
        if self.transform:
            # undo normalization from N(0, 1) back to [0, 1]
            imgs = self.invert_transform_func(imgs)
            # get max values for every data modality
            max_vals = np.array(
                list(itertools.chain(
                    *[self.stats[name]["max"] for name in self.data_order]
                ))).reshape(
                    -1, 1, 1)
            # invert scaling from [0, 1] to the original data scale
            imgs = imgs * max_vals
        # permute dimensions from channels x height x width to height x width
        # x channels
        imgs = imgs.permute(1, 2, 0).cpu().numpy()
        return imgs

    def get_imgs(self, left: float, bottom: float, right: float, top: float,
                 country: str, output_size: Dict = OUTPUT_SIZE) -> np.ndarray:
        """Extract image from rasterized images based on bounds.

        Args:
            left (float): Longitude.
            bottom (float): Latitude.
            right (float): Longitude.
            top (float): Latitude.
            output_size (Dict): Dictionary with output size.
        Returns:
            imgs (np.ndarray): An array of size width x height x channels.
        """
        imgs = []
        for data_name in self.data_order:
            raster = self.data_rasters[country][data_name]
            window = windows.from_bounds(
                left, bottom, right, top, raster.transform)
            for c in self.train_metadata[
                    country][data_name]["raster_channels"]:
                # read raster at tile and resize to output size
                r = raster.read(
                    c, window=window, out_shape=output_size[self.tile_size])
                # if any values are greater than 1., set to zero
                if data_name == "admin_bounds":
                    r[r > 1.] = 0.
                # if any values are smaller than 0., set to zero
                elif data_name == "population":
                    r[r < 0] = 0
                imgs.append(np.expand_dims(r, -1))
        imgs = np.concatenate(imgs, -1)
        return imgs

    def transform_imgs(self, imgs: torch.Tensor) -> torch.Tensor:
        """Normalize image according to pre-calculated statistics.

        Args:
            img (torch.Tensor): An array with shape channels x height x width.

        Returns:
            img (torch.Tensor): An array with shape channels x height x width
                normalized.
        """
        # max values of shape (#data modalities, 1, 1)
        max_vals = np.array(
            list(itertools.chain(
                *[self.stats[name]["max"] for name in self.data_order]
            ))).reshape(-1, 1, 1)
        # scale to 0 and 1
        imgs = imgs / max_vals
        # transform
        imgs = self.transform_func(imgs)
        return imgs

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor]:
        """Get images and label based on index"""
        # get dataset entry
        entry = self.train_gdf.iloc[idx]
        country = entry.Country
        # positive (= bridge site)
        if entry.pos_neg == "pos":
            label = 1
            # get longitude and latitude of bridge site as center point for the
            # tile
            lon, lat = entry["GPS (Longitude)"], entry["GPS (Latitude)"]
            if self.use_rnd_center_point:
                # shift center point
                lon, lat = shift_coords(lon, lat)
                # check whether the center point is a valid point within the
                # entry's bounds
                if not is_valid_lonlat(entry, lon, lat, self.tile_size):
                    lon, lat = entry[
                        "GPS (Longitude)"], entry["GPS (Latitude)"]
        # negative (= no bridge site)
        elif entry.pos_neg == "neg":
            label = 0
            valid_point = False
            # try to sample 5 times
            max_num_tries, num_tries = 5, 0
            # not fail save
            while num_tries < max_num_tries and valid_point is False:
                num_tries += 1
                # sample a random center point within the negative area
                lon, lat = sample_points_in_polygon(entry.geometry)[0]
                # check whether the center point is a valid point within the
                # entry's bounds
                if is_valid_lonlat(entry, lon, lat, self.tile_size):
                    valid_point = True
        else:
            raise NotImplementedError

        # get tile bounds
        left, bottom, right, top = get_tile_bounds(lon, lat, self.tile_size)
        # get images based on tile bounds
        imgs = self.get_imgs(left, bottom, right, top, country)
        # use augmentation
        if self.use_augment:
            imgs = self.augment(imgs)
        # imgs of shape (#data modalities, h, w)
        imgs = torch.from_numpy(imgs.copy()).permute(2, 0, 1)
        # transform imgs
        if self.transform:
            imgs = self.transform_imgs(imgs)
        return imgs.float(), label

    def __len__(self) -> int:
        """Returns length of the dataset"""
        return len(self.train_gdf)

    def augment(self, imgs: np.ndarray, random_admin: float = 0.75,
                random_ele: Tuple[float] = (0, 2), random_fliplr: float = 0.5,
                random_osm: float = 0.75,
                random_osm_img: Tuple[float] = (0, 3),
                random_pop: Tuple[float] = (0, 3),
                random_slo: Tuple[float] = (0, 2)) -> np.ndarray:
        """
        Augments the images.

        Args:
            img (np.ndarray): An array with shape height x width x channels.
            random_admin (float, optional): Probability of not setting admin
                boundaries (pixel values set to 1) to 0. Default: 0.75.
            random_ele (Tuple[float], optional): Mean and standard deviation
                for augmentation of elevation values. Default: (0, 2).
            random_osm (float, optional): Probability of not setting OSM label
                (pixel values set to 1) to 0. Default: 0.75.
            random_osm_img (Tuple[float], optional): Mean and standard
                deviation for augmenting OSM image values. Default: (0, 3).
            random_pop (Tuple[float], optional): Mean and standard
                deviation for augmenting population data. Default: (0, 3).
            random_fliplr (float, optional): Probability of not flipping the
                images horizontally. Default: 0.5.

        Returns:
            img (np.ndarray): An array with shape height x width x channels
                that was augmented.
        """
        if np.random.rand() > random_fliplr:
            # flip image horizontally
            imgs = np.fliplr(imgs)

        INDECES = [0]
        current_ind = 0
        for i in self.data_order[:-1]:
            current_ind += len(METADATA["Rwanda"][i]["raster_channels"])
            INDECES.append(current_ind)

        if "population" in self.data_order:
            # augment population
            imgs = augment_pop_data(
                imgs, INDECES[self.data_order.index("population")],
                self.stats["population"], loc=random_pop[0],
                scale=random_pop[1])

        if "elevation" in self.data_order:
            # augment elevation
            imgs = augment_terrain_data(
                imgs, INDECES[self.data_order.index(
                    "elevation")], loc=random_ele[0], scale=random_ele[1])

        if "slope" in self.data_order:
            # augment slope
            imgs = augment_terrain_data(
                imgs, INDECES[self.data_order.index("slope")],
                loc=random_slo[0], scale=random_slo[1])

        # augment osm img
        if "osm_img" in self.data_order:
            augment_osm_img(
                imgs, INDECES[self.data_order.index("osm_img")],
                loc=random_osm_img[0], scale=random_osm_img[1])

        if "roads" in self.data_order:
            # augment roads
            imgs = augment_binary_map(
                imgs, INDECES[self.data_order.index("roads")],
                p_reject=random_osm)

        if "waterways" in self.data_order:
            # augment waterways
            imgs = augment_binary_map(
                imgs, INDECES[self.data_order.index("waterways")],
                p_reject=random_osm)

        if "admin_bounds" in self.data_order:
            # augment admin boundaries
            imgs = augment_binary_map(
                imgs, INDECES[self.data_order.index("admin_bounds")],
                p_reject=random_admin)

        return imgs


class TestBridgeDataset(BridgeDataset):
    def __init__(self, data: Dict = TRAIN_DATA,
                 data_order: List[str] = DATA_ORDER, data_version: str = "v1",
                 num_test_samples: int = 8, raster_data: str = METADATA,
                 stats_fp: str = STATS_FP, tile_size: int = 300,
                 transform: bool = True, use_augment: bool = True,
                 use_rnd_center_point: bool = True) -> None:
        """
        Args:
            data (Dict, optional): Dictionary contains the train data file path
                for the `data_version` and `tile_size`. Default: TRAIN_DATA.
            data_order (str, optional): The order of the data modalities to be
                loaded and read. Default: DATA_ORDER.
            data_version (str, optional): The version of the data, can be
                either `v1` or `v2`. Default: v1.
            num_test_samples: How many test samples per tile are used during
                test time.
            raster_data (Dict, optional): This dictionary contains for each
                country the data modality with its data path and the target
                channels that can be read with rasterio. Default: METADATA.
            stats_fp (str, optional): The file path of the data stats. Default:
                STATS_FP.
            tile_size (int, optional): Tile (square) size, can be either 300,
                600, or 1200 metres. Default: 300.
            transform (bool, optional): Whether to normalize the data or not.
                Default: True.
            use_augment (bool, optional): Whether to use augmentation on the
                 data or not. Default: True.
            use_rnd_center_point (bool, optional): Whether to use random tile
                center points or not. Default: True.
        """
        super(TestBridgeDataset, self).__init__(
            data=data, data_order=data_order, data_version=data_version,
            raster_data=raster_data, stats_fp=stats_fp,
            tile_size=tile_size, use_rnd_center_point=use_rnd_center_point,
            transform=transform, use_augment=use_augment)
        self.num_test_samples = num_test_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor]:
        """Returns `num_test_samples` samples of each entry `idx`."""
        # get dataset entry
        entry = self.train_gdf.iloc[idx]
        # positive (= bridge site)
        if entry.pos_neg == "pos":
            images = []
            # get `num_test_samples` samples
            for _ in range(self.num_test_samples):
                image, label = super().__getitem__(idx)
                images.append(image.unsqueeze(0))
            # concatenate all tile images, shape is (`num_test_samples`,
            # `channels`, `width`, `height`)
            images = torch.cat(images, 0)
        # negatie (= no bridge site)
        else:
            label = 0
            images = []
            # sample a center point
            lon, lat = sample_points_in_polygon(entry.geometry)[0]
            lon_lat_list = [(lon, lat)]
            # shift center point to have several test samples
            # (= num_test_samples) and add to `lon_lat_list`
            while len(lon_lat_list) < self.num_test_samples:
                lon_, lat_ = shift_coords(lon, lat)
                point = Point(lon_, lat_)
                if entry.geometry.contains(point):
                    lon_lat_list.append((lon_, lat_))
            # for each point in `lon_lat_list`
            for lon, lat in lon_lat_list:
                # get tile bounds
                left, bottom, right, top = get_tile_bounds(
                    lon, lat, self.tile_size)
                # get images from the bounds
                imgs = self.get_imgs(left, bottom, right, top, entry.Country)
                # use augmentation
                if self.use_augment:
                    imgs = self.augment(imgs)
                # convert to torch.Tensor and permute to
                # (#data modalities, h, w)
                imgs = torch.from_numpy(imgs.copy()).permute(2, 0, 1)
                # normalize images
                if self.transform:
                    imgs = self.transform_imgs(imgs)
                images.append(imgs.unsqueeze(0))
            images = torch.cat(images, 0)

        return images.float(), label


class BridgeSampler(Sampler[int]):
    """Samples positive and negative samples from bridge dataset."""

    def __init__(self, tile_size: int, data: Dict = TRAIN_DATA,
                 data_version: str = "v1", num_samples: int = None,
                 set_name: str = "train", shuffle: bool = True) -> None:
        """Samples positive and negative tiles from bridge train dataset.

        Args:
            tile_size (int): Tile (square) size, can be either 300, 600,
                or 1200 metres.
            data (str, optional): Dictionary contains the train data file path
                for the `data_version` and `tile_size`. Default: TRAIN_DATA.
            data_version (str, optional): The version of the data, can be
                either `v1` or `v2`. Default: v1.
            num_samples (int, optional): Number of total samples that this
                sampler should output. If num_samples is None, the the
                num_samples is double the number of positive samples. Default:
                None.
            set_name (str, optional): Name of the set that is used. Can be
                either `train`, `val` or `test`. Default: train.
            shuffle (bool, optional): Whether to shuffle the samples or not.
                Default: True.
        """
        assert tile_size in [300, 600, 1200], "Tile size not known"
        assert set_name in ["train", "val", "test"], "Set name not known."
        assert data_version in data and tile_size in data[data_version], \
            "Expected for data[data_version][tile_size] to exist."

        self.shuffle = shuffle

        # open training data
        with open(data[data_version][tile_size]) as f:
            self.train_gdf = gpd.read_file(f)

        # get number of positive data samples
        self.num_pos_samples = len(self.train_gdf[
            self.train_gdf.split.str.startswith(set_name) &
            (self.train_gdf.pos_neg == "pos")
        ])
        # if num_samples (total number of data samples) is None, it will be set
        # to double the number of positive data samples
        if num_samples is None:
            self.num_samples = 2 * self.num_pos_samples
        else:
            assert isinstance(num_samples, int), \
                "Expected num_samples to be int"
            # num_samples is at least larger than number of positive data
            # samples to include at least 1 negative sample
            self.num_samples = max(num_samples, self.num_pos_samples + 1)

        # positive samples with specific set name `set_name`
        pos = self.train_gdf[
            self.train_gdf.split.str.startswith(set_name) & (
                self.train_gdf.pos_neg == "pos")].index.tolist()
        # negative samples with specific set name `set_name`
        neg = self.train_gdf[
            self.train_gdf.split.str.startswith(set_name) & (
                self.train_gdf.pos_neg == "neg")].index.tolist()

        # number of negatives
        num_neg = max(self.num_samples - len(pos), len(neg))
        # number of samples per negative if the number of samples is larger
        # than the number of negatives
        num_neg_per_neg = num_neg // len(neg)
        new_neg = []
        # make a list of negatives (= indeces of negative entries)
        # for each duplicate the negatives by the number of samples
        # `num_neg_per_neg` above
        for i in range(len(neg) - 1):
            new_neg += [neg[i]] * num_neg_per_neg
        new_neg += [neg[-1]] * (num_neg - len(new_neg))
        neg = new_neg

        # all indeces, both positive and negative
        self.indeces = list(pos + neg)

    def __iter__(self) -> Iterator[int]:

        # shuffle list of negative and positives
        if self.shuffle:
            ind_order = np.random.choice(
                list(range(len(self.indeces))), len(self.indeces),
                replace=False)
        else:
            ind_order = range(len(self.indeces))
        # return indeces
        indeces = [self.indeces[i] for i in ind_order]
        yield from iter(indeces)

    def __len__(self) -> int:
        return self.num_samples


class NoLabelTileDataset(BridgeDataset):

    def __init__(self, data: Dict = TRAIN_DATA,
                 data_order: List[str] = DATA_ORDER, data_version: str = "v1",
                 len_dataset: int = 1000, num_samples: int = 2,
                 raster_data: str = METADATA, stats_fp: str = STATS_FP,
                 tile_size: int = 300, transform: bool = True,
                 use_augment: bool = True,
                 use_rnd_center_point: bool = True) -> None:
        """
        data (Dict, optional): Dictionary contains the train data file path for
            the `data_version` and `tile_size`. Default: TRAIN_DATA.
        data_order (str, optional): The order of the data modalities to be
            loaded and read. Default: DATA_ORDER.
        data_version (str, optional): The version of the data, can be either
            `v1` or `v2`. Default: v1.
        len_dataset: Length of the dataset.
        num_samples: How many test samples per tile are used during
            test time.
        raster_data (Dict, optional): This dictionary contains for each
                country the data modality with its data path and the target
                channels that can be read with rasterio. Default: METADATA.
        stats_fp (str, optional): The file path of the data stats. Default:
            STATS_FP.
        tile_size (int, optional): Tile (square) size, can be either 300, 600,
            or 1200 metres. Default: 300.
        transform (bool, optional): Whether to normalize the data or not.
            Default: True.
        use_augment (bool, optional): Whether to use augmentation on the data
            or not. Default: True.
        use_rnd_center_point (bool, optional): Whether to use random tile
            center points or not. Default: True.
        """
        super(NoLabelTileDataset, self).__init__(
            data=data, data_order=data_order, data_version=data_version,
            raster_data=raster_data, stats_fp=stats_fp,
            tile_size=tile_size, use_rnd_center_point=use_rnd_center_point,
            transform=transform, use_augment=use_augment)

        self.len_dataset = len_dataset
        self.num_samples = num_samples
        self.train_gdf = self.train_gdf[self.train_gdf.id.str.contains("ssl")]
        self.train_gdf = self.train_gdf.reset_index(drop=True)
        assert len(self.train_gdf) == 1, "Expected dataset length to be one"

    def __len__(self) -> int:
        return self.len_dataset

    def __getitem__(self, _) -> torch.Tensor:
        # get entry
        entry = self.train_gdf.iloc[0]
        # sample a point within the geometry
        lon, lat = sample_points_in_polygon(
            entry.geometry, add_padding=True)[0]
        lon_lat_list = [(lon, lat)]
        # sample until we have `num_samples` points
        while len(lon_lat_list) < self.num_samples:
            lon_, lat_ = shift_coords(lon, lat)
            point = Point(lon_, lat_)
            if entry.geometry.contains(point):
                lon_lat_list.append((lon_, lat_))
        # get images for the list of points
        images = []
        for lon, lat in lon_lat_list:
            # get bounds
            left, bottom, right, top = get_tile_bounds(
                lon, lat, self.tile_size)
            # get images
            try:
                imgs = self.get_imgs(left, bottom, right, top, entry.Country)
            except Exception as e:
                print(e)
                print(left, bottom, right, top)
            # augment
            if self.use_augment:
                imgs = self.augment(imgs)
            # create torch.Tensor and permute dimensions from (width, height,
            # channels) to channels, width, height
            imgs = torch.from_numpy(imgs.copy()).permute(2, 0, 1)
            # normalizes
            if self.transform:
                imgs = self.transform_imgs(imgs)
            images.append(imgs.unsqueeze(0))
        # (num_samples, channels, width, height)
        images = torch.cat(images, 0)

        return images.float()


def worker_init_fn(worker_id):
    """Set seed for dataloader"""
    np.random.seed(np.random.get_state()[1][0] + worker_id)


def get_dataloaders(batch_size: int, tile_size: int,
                    data_version: str = "v1", data_order: List = DATA_ORDER,
                    num_test_samples: int = 64, num_workers: int = 0,
                    stats_fp: str = STATS_FP, test_batch_size: int = 10,
                    transform: bool = True, train_data: str = TRAIN_DATA,
                    train_metadata: str = METADATA, use_augment: bool = True,
                    use_rnd_center_point: bool = True,
                    use_several_test_samples: bool = False) -> Tuple[
                        DataLoader, DataLoader]:
    """Returns dataloaders for training and evaluation.

        Args:
            data_order (str, optional): The order of the data modalities to be
                loaded and read. Default: DATA_ORDER.
            data_version (str, optional): The version of the data, can be
                either `v1` or `v2`. Default: v1.
            num_test_samples: How many test samples per tile are used during
                test time.
            test_batch_size (int, optional): Batch size for the test
                dataloaders. Default: 10.
            stats_fp (str, optional): The file path of the data stats. Default:
                STATS_FP.
            tile_size (int, optional): Tile (square) size, can be either 300,
                600, or 1200 metres. Default: 300.
            train_data (str, optional): Dictionary contains the train data file
                path for the `data_version` and `tile_size`. Default:
                TRAIN_DATA.
            train_metadata (str, optional): Default: METADATA.
            transform (bool, optional): Whether to normalize the data or not.
                Default: True.
            use_augment (bool, optional): Whether to use augmentation on the
                 data or not. Default: True.
            use_rnd_center_point (bool, optional): Whether to use random tile
                center points or not. Default: True.
            use_several_test_samples (bool, optional). Whether to use several
                samples for testing or not. Default: False.

        Returns:
            dataloader_train (DataLoader): Train dataloader.
            dataloader_validation (DataLoader): Validation dataloader.
            dataloader_test (DataLoader): Test dataloader.
            dataloader_nolab (DataLoader): No labelled dataloader.

    """
    # train dataset
    tr_dataset = BridgeDataset(
        data=train_data, data_order=data_order, data_version=data_version,
        raster_data=train_metadata, stats_fp=stats_fp, tile_size=tile_size,
        transform=transform, use_augment=use_augment,
        use_rnd_center_point=True)
    # validation and test dataset
    if use_several_test_samples:
        va_te_dataset = TestBridgeDataset(
            data=train_data, data_order=data_order, data_version=data_version,
            num_test_samples=num_test_samples, raster_data=train_metadata,
            stats_fp=stats_fp, tile_size=tile_size, transform=transform,
            use_augment=use_augment, use_rnd_center_point=True)
    else:
        va_te_dataset = BridgeDataset(
            data=train_data, data_order=data_order, data_version=data_version,
            raster_data=train_metadata, stats_fp=stats_fp, tile_size=tile_size,
            transform=transform, use_augment=False,
            use_rnd_center_point=True)

    # samplers
    common_sampler_kwargs = {
        "tile_size": tile_size,
        "data": train_data,
        "data_version": data_version
    }
    sampler_train = BridgeSampler(
        set_name="train", shuffle=True, **common_sampler_kwargs)
    sampler_validation = BridgeSampler(
        set_name="val", shuffle=False, **common_sampler_kwargs)
    sampler_test = BridgeSampler(
        set_name="test", shuffle=False, **common_sampler_kwargs)

    # dataloaders
    common_loader_kwargs = {
        "worker_init_fn": worker_init_fn,
        "num_workers": num_workers
    }
    te_batch_size = test_batch_size if use_several_test_samples else batch_size
    dataloader_train = DataLoader(
        tr_dataset, sampler=sampler_train, batch_size=batch_size,
        drop_last=True, **common_loader_kwargs)
    dataloader_validation = DataLoader(
        va_te_dataset, sampler=sampler_validation, batch_size=te_batch_size,
        **common_loader_kwargs)
    dataloader_test = DataLoader(
        va_te_dataset, sampler=sampler_test, batch_size=te_batch_size,
        **common_loader_kwargs)

    # unlabelled dataset
    uganda_dataset = NoLabelTileDataset(
        data=train_data, data_order=data_order, data_version=data_version,
        len_dataset=2000, raster_data=train_metadata,
        stats_fp=stats_fp, tile_size=tile_size, transform=transform,
        use_augment=use_augment, use_rnd_center_point=use_rnd_center_point
    )
    dataloader_nolab = DataLoader(
        uganda_dataset, batch_size=batch_size, drop_last=True,
        **common_loader_kwargs
    )

    return (
        dataloader_train, dataloader_validation, dataloader_test,
        dataloader_nolab)
