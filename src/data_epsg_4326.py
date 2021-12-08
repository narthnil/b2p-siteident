import numpy as np
import geopandas as gpd
import rasterio

from rasterio import windows
from typing import Iterator

from geographiclib.geodesic import Geodesic
from shapely.geometry import Point

from torch.utils.data import Dataset, Sampler


TRAIN_METADATA = {
    "Rwanda": {
        "population": {
            "fp": ("./data/population/"
                   "Rwanda_population_data_2020_UNadj_constrained.tif"),
            "raster_channels": [1]
        },
        "osm_imgs": {
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
        }
    },
    "Uganda": {
        "population": {
            "fp": ("./data/population/"
                   "Uganda_population_data_2020_UNadj_constrained.tif"),
            "raster_channels": [1]
        },
        "osm_imgs": {
            "fp": "./data/osm/imgs/uganda_train_osm_nolab_1-50000_4326.tiff",
            "raster_channels": [1, 2, 3]
        },
        "elevation": {
            "fp": "./data/slope_elevation/elevation_uganda.tif",
            "raster_channels": [2]
        },
        "slope": {
            "fp": "./data/slope_elevation/slope_uganda.tif",
            "raster_channels": [2]
        }
    },
}

OUTPUT_SIZE = {300: (12, 12), 600: (24, 24), 1200: (48, 48)}
DATA_ORDER = ["population", "osm_imgs", "elevation", "slope"]

TRAINING_DATA = {
    300: "./data/ground_truth/v2/training_data_300.csv",
    600: "./data/ground_truth/v2/training_data_600.csv",
    1200: "./data/ground_truth/v2/training_data_1200.csv"
}


def get_square_area(longitude: float, latitude: float,
                    square_length: float = 60., geod: Geodesic = None):
    """
    Args:
        lat: Latitude
        lon: Longtitude
        square_length: Measured in meters.
        geod: (Geodesic)
    """
    if geod is None:
        # Define the ellipsoid
        geod = Geodesic.WGS84

    diag_len = np.sqrt(2 * (square_length / 2.)**2)
    lr_point = geod.Direct(latitude, longitude, 135, diag_len)
    ll_point = geod.Direct(latitude, longitude, -135, diag_len)
    ul_point = geod.Direct(latitude, longitude, -45, diag_len)
    ur_point = geod.Direct(latitude, longitude, 45, diag_len)
    return [
        # lr = lower right
        [lr_point['lon2'], lr_point['lat2']],
        # ll = lower left
        [ll_point['lon2'], ll_point['lat2']],
        # ul = upper left
        [ul_point['lon2'], ul_point['lat2']],
        # ur = upper right
        [ur_point['lon2'], ur_point['lat2']]
    ]


class BridgeDataset(Dataset):
    def __init__(self, tile_size=300, use_rnd_pos=False, transform=False):
        assert tile_size in [300, 600, 1200], "Tile size not known"

        self.tile_size = tile_size
        with open("./data/ground_truth/v2/train_{}.geojson".format(
                tile_size)) as f:
            self.train_gdf = gpd.read_file(f)
        self.data_rasters = {}
        for country, data_modalities in TRAIN_METADATA.items():
            if country not in self.data_rasters:
                self.data_rasters[country] = {}
            for data_type, data in data_modalities.items():
                self.data_rasters[country][data_type] = rasterio.open(
                    data["fp"])

        self.use_rnd_pos = use_rnd_pos
        self.transform = transform

    @staticmethod
    def shift_coords(lon, lat, tile_size=300):
        geod = Geodesic.WGS84
        lat_shift, lon_shift = np.clip(
            np.random.normal(
                loc=0.0, scale=(tile_size - 50) / 4, size=2
            ),
            - (tile_size - 50) / 2, (tile_size - 50) / 2).tolist()
        if lat_shift < 0:
            lat_shift_degree = 180
        else:
            lat_shift_degree = 0
        if lon_shift < 0:
            lon_shift_degree = 90
        else:
            lon_shift_degree = -90
        lat_shifted = geod.Direct(lat, lon, lat_shift_degree, lat_shift)
        new_lat, new_lon = lat_shifted["lat2"], lat_shifted["lon2"]
        lon_shifted = geod.Direct(
            new_lat, new_lon, lon_shift_degree, lat_shift)
        new_lat, new_lon = lon_shifted["lat2"], lon_shifted["lon2"]
        return new_lon, new_lat

    @staticmethod
    def sample_points_in_polygon(polygon, num_samples=1):
        points = []
        min_x, min_y, max_x, max_y = polygon.bounds
        while len(points) < num_samples:
            x, y = np.random.uniform(
                min_x, max_x), np.random.uniform(min_y, max_y)
            point = Point(x, y)
            if polygon.contains(point):
                points.append((x, y))
        return points

    def __getitem__(self, idx):
        # get dataset entry
        entry = self.train_gdf.iloc[idx]
        country = entry.Country
        # positives
        if entry.pos_neg == "pos":
            label = 1
            lon, lat = entry["GPS (Longitude)"], entry["GPS (Latitude)"]
            if self.use_rnd_pos:
                valid_point = False
                max_num_tries, num_tries = 5, 0
                while num_tries < max_num_tries or valid_point is False:
                    num_tries += 1
                    lon, lat = self.shift_coords(lon, lat)
                    area_coords = get_square_area(
                        lon, lat, square_length=self.tile_size)
                    left = min([ac[0] for ac in area_coords])
                    right = max([ac[0] for ac in area_coords])
                    if (entry.min_x < left < entry.max_x and
                            entry.min_x < right < entry.max_x):
                        valid_point = True
            if valid_point is False:
                lon, lat = entry["GPS (Longitude)"], entry["GPS (Latitude)"]
        elif entry.pos_neg == "neg":
            label = 0
            valid_point = False
            max_num_tries, num_tries = 5, 0
            while num_tries < max_num_tries or valid_point is False:
                num_tries += 1
                lon, lat = self.sample_points_in_polygon(entry.geometry)[0]
                area_coords = get_square_area(
                    lon, lat, square_length=self.tile_size)
                left = min([ac[0] for ac in area_coords])
                right = max([ac[0] for ac in area_coords])
                if (entry.min_x < left < entry.max_x and
                        entry.min_x < right < entry.max_x):
                    valid_point = True
        else:
            raise NotImplementedError

        # coordinates of area with size tile_size
        area_coords = get_square_area(
            lon, lat, square_length=self.tile_size)
        # get left = lat, bottom = lon, right = lat, top = lon
        left = min([ac[0] for ac in area_coords])
        bottom = min([ac[1] for ac in area_coords])
        right = max([ac[0] for ac in area_coords])
        top = max([ac[1] for ac in area_coords])

        imgs = []
        for data_name in DATA_ORDER:
            raster = self.data_rasters[country][data_name]
            window = windows.from_bounds(
                left, bottom, right, top, raster.transform)
            for c in TRAIN_METADATA[country][data_name]["raster_channels"]:
                r = raster.read(
                    c, window=window, out_shape=OUTPUT_SIZE[self.tile_size])
                imgs.append(np.expand_dims(r, -1))
        # TODO transform to torch.Tensor
        # TODO return label
        return np.abs(np.concatenate(imgs, -1)), label

    def __len__(self):
        return len(self.train_gdf)


class BridgeSampler(Sampler[int]):

    def __init__(self, tile_size, num_samples=None, set_name="train",
                 shuffle=True) -> None:
        assert tile_size in [300, 600, 1200], "Tile size not known"
        assert set_name in ["train", "val", "test"], "Set name not known."

        with open("./data/ground_truth/v2/train_{}.geojson".format(
                tile_size)) as f:
            train_gdf = gpd.read_file(f)

        self.num_pos_samples = len(train_gdf[
            train_gdf.split.str.startswith(set_name) &
            (train_gdf.pos_neg == "pos")
        ])
        if num_samples is None:
            self.num_samples = 2 * self.num_pos_samples
        else:
            assert isinstance(num_samples, int), \
                "Expected num_samples to be int"
            self.num_samples = max(num_samples, self.num_pos_samples)
        self.shuffle = shuffle
        self.train_gdf = train_gdf

        pos = train_gdf[
            train_gdf.split.str.startswith(set_name) & (
                train_gdf.pos_neg == "pos")
        ].index.tolist()

        neg = train_gdf[
            train_gdf.split.str.startswith(set_name) & (
                train_gdf.pos_neg == "neg")
        ].index.tolist()

        num_neg = max(self.num_samples - len(pos), len(neg))
        num_neg_per_neg = num_neg // len(neg)
        new_neg = []

        for i in range(len(neg) - 1):
            new_neg += [neg[i]] * num_neg_per_neg
        new_neg += [neg[-1]] * (num_neg - len(new_neg))
        neg = new_neg

        self.indeces = list(pos + neg)

    def __iter__(self) -> Iterator[int]:

        if self.shuffle:
            ind_order = np.random.choice(
                list(range(len(self.indeces))), len(self.indeces),
                replace=False)
        else:
            ind_order = range(len(self.indeces))
        indeces = [self.indeces[i] for i in ind_order]
        yield from iter(indeces)

    def __len__(self) -> int:
        return self.num_samples
