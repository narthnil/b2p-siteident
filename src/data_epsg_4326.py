import numpy as np
import pandas as pd
import rasterio

from rasterio import windows

from geographiclib.geodesic import Geodesic

from torch.utils.data import Dataset


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
    300: "./data/ground_truth/training_data_300.csv",
    600: "./data/ground_truth/training_data_600.csv",
    1200: "./data/ground_truth/training_data_1200.csv"
}
DTYPE = {
    "Opportunity ID": str,
    "GPS (Latitude)": float,
    "GPS (Longitude)": float,
    "Country": str,
    "Split": str,
    "left": float,
    "right": float,
    "bottom": float,
    "top": float,
    "pos_neg": str
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
        self.training_data = pd.read_csv(
            TRAINING_DATA[tile_size],
            dtype=DTYPE
        )
        self.data_rasters = {}
        for country, data_modalities in TRAIN_METADATA.items():
            if country not in self.data_rasters:
                self.data_rasters[country] = {}
            for data_type, data in data_modalities.items():
                self.data_rasters[country][data_type] = rasterio.open(
                    data["fp"])

        self.geod = Geodesic.WGS84
        self.use_rnd_pos = use_rnd_pos
        self.transform = transform

    def shift_coords(self, lon, lat):
        lat_shift, lon_shift = np.clip(
            np.random.normal(
                loc=0.0, scale=(self.tile_size - 50) / 4, size=2
            ),
            - (self.tile_size - 50) / 2,
            (self.tile_size - 50) / 2
        ).tolist()
        if lat_shift < 0:
            lat_shift_degree = 180
        else:
            lat_shift_degree = 0
        if lon_shift < 0:
            lon_shift_degree = 90
        else:
            lon_shift_degree = -90
        lat_shifted = self.geod.Direct(lat, lon, lat_shift_degree, lat_shift)
        new_lat, new_lon = lat_shifted["lat2"], lat_shifted["lon2"]
        lon_shifted = self.geod.Direct(
            new_lat, new_lon, lon_shift_degree, lat_shift)
        new_lat, new_lon = lon_shifted["lat2"], lon_shifted["lon2"]
        return new_lon, new_lat

    def __getitem__(self, idx):
        # get dataset entry
        entry = self.training_data.iloc[idx]
        country = entry.Country
        # positives
        if entry.pos_neg == "pos":
            label = 1
            lon, lat = entry["GPS (Longitude)"], entry["GPS (Latitude)"]
            if self.use_rnd_pos:
                lon, lat = self.shift_coords(lon, lat)
            # coordinates of area with size tile_size
            area_coords = get_square_area(
                lon, lat, square_length=self.tile_size)
            # get left = lat, bottom = lon, right = lat, top = lon
            left = min([ac[0] for ac in area_coords])
            bottom = min([ac[1] for ac in area_coords])
            right = max([ac[0] for ac in area_coords])
            top = max([ac[1] for ac in area_coords])
        elif entry.pos_neg == "neg":
            label = 0
            left = entry.left
            right = entry.right
            bottom = entry.bottom
            top = entry.top
        else:
            raise NotImplementedError
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
        return np.abs(np.concatenate(imgs, -1))

    def __len__(self):
        return len(self.training_data)
