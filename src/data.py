import json
import itertools

from typing import Tuple

import numpy as np
import geopandas as gpd
import rasterio

from rasterio import windows
from typing import Iterator, Tuple

from geographiclib.geodesic import Geodesic
from shapely.geometry import Point
from torchvision import transforms

import torch
from torch.utils.data import Dataset, Sampler, DataLoader

OUTPUT_SIZE = {300: (12, 12), 600: (24, 24), 1200: (48, 48)}
DATA_ORDER = ["population", "osm_imgs", "elevation", "slope", "roads",
              "waterways", "admin_bounds"]

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
        },
        "roads": {
            "fp": "./data/osm/roads/rwanda-osm-roads.tif",
            "raster_channels": [1]
        },
        "waterways": {
            "fp": "./data/osm/waterways/rwanda-osm-waterways.tif",
            "raster_channels": [1]
        },
        "admin_bounds": {
            "fp": ("./data/admin_boundaries/"
                   "Rwanda_Village_AdminBoundary_1_3600_clipped.tiff"),
            "raster_channels": [1]
        }
    },
    "Uganda": {
        "population": {
            "fp": ("./data/population/"
                   "Uganda_population_data_2020_UNadj_constrained.tif"),
            "raster_channels": [1]
        },
        "osm_imgs": {
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
        "admin_bounds": {
            "fp": ("./data/admin_boundaries/"
                   "Uganda_Parish_AdminBoundary_1_3600_clipped.tiff"),
            "raster_channels": [1]
        }
    },
}

TRAIN_DATA = {
    300: "./data/ground_truth/train_300.geojson",
    600: "./data/ground_truth/train_600.geojson",
    1200: "./data/ground_truth/train_1200.geojson"
}

STATS_FP = "./data/ground_truth/stats.json"


class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        i = 0
        for t, m, s in zip(tensor, self.mean, self.std):
            t.sub_(m).div_(s)
            i += 1
        return tensor


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
    def __init__(self, tile_size: int = 300, use_rnd_pos: bool = False,
                 transform: bool = False, train_data: str = TRAIN_DATA,
                 train_metadata: str = TRAIN_METADATA, stats_fp=STATS_FP,
                 use_augment: bool = True):
        assert tile_size in [300, 600, 1200], "Tile size not known"

        self.tile_size = tile_size
        with open(train_data[tile_size]) as f:
            self.train_gdf = gpd.read_file(f)
        self.data_rasters = {}
        for country, data_modalities in train_metadata.items():
            if country not in self.data_rasters:
                self.data_rasters[country] = {}
            for data_type, data in data_modalities.items():
                self.data_rasters[country][data_type] = rasterio.open(
                    data["fp"])

        self.use_rnd_pos = use_rnd_pos
        self.transform = transform
        self.use_augment = use_augment

        with open(stats_fp) as f:
            self.stats = json.load(f)
        self.transform_func = transforms.Compose([
            Normalize(
                list(itertools.chain(
                    *[self.stats[name]["mean"] for name in DATA_ORDER])),
                list(itertools.chain(
                    *[self.stats[name]["std"] for name in DATA_ORDER])))

        ])
        self.invert_transform_func = transforms.Compose([
            UnNormalize(
                list(itertools.chain(
                    *[self.stats[name]["mean"] for name in DATA_ORDER])),
                list(itertools.chain(
                    *[self.stats[name]["std"] for name in DATA_ORDER])))

        ])
        self.train_metadata = train_metadata

    def convert_tensor_2_numpy(self, imgs):
        if self.transform:
            imgs = self.invert_transform_func(imgs)
            max_vals = np.array(
                list(itertools.chain(
                    *[self.stats[name]["max"] for name in DATA_ORDER]
                ))).reshape(
                    -1, 1, 1)
            imgs = imgs * max_vals
        imgs = imgs.permute(1, 2, 0).cpu().numpy()
        return imgs

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
            valid_point = True
            if self.use_rnd_pos:
                valid_point = False
                max_num_tries, num_tries = 5, 0
                # not fail save
                while num_tries < max_num_tries and valid_point is False:
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
            # not fail save
            while num_tries < max_num_tries and valid_point is False:
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
            for c in self.train_metadata[
                    country][data_name]["raster_channels"]:
                r = raster.read(
                    c, window=window, out_shape=OUTPUT_SIZE[self.tile_size])
                if data_name == "admin_bounds":
                    r[r == 127.] = 0.
                elif data_name == "population":
                    r[r < 0] = 0
                imgs.append(np.expand_dims(r, -1))

        imgs = np.concatenate(imgs, -1)
        if self.use_augment:
            imgs = self.augment(imgs)
        # (#data modalities, h, w)
        imgs = torch.from_numpy(imgs.copy()).permute(2, 0, 1)
        if self.transform:
            # of shape (#data modalities, 1, 1)
            max_vals = np.array(
                list(itertools.chain(
                    *[self.stats[name]["max"] for name in DATA_ORDER]
                ))).reshape(
                    -1, 1, 1)
            imgs = imgs / max_vals
            imgs = self.transform_func(imgs)
        return imgs, label

    def __len__(self):
        return len(self.train_gdf)

    def augment(self, img, random_fliplr: float = 0.5,
                random_pop: Tuple = (0, 3), random_ele: Tuple = (0, 2),
                random_slo: Tuple = (0, 2), random_osm_imgs: Tuple = (0, 3),
                random_osm: float = 0.75):
        if np.random.rand() > random_fliplr:
            img = np.fliplr(img)

        INDECES = [0]
        current_ind = 0
        for i in DATA_ORDER[:-1]:
            current_ind += len(TRAIN_METADATA["Rwanda"][i]["raster_channels"])
            INDECES.append(current_ind)

        index_pop = INDECES[DATA_ORDER.index("population")]
        height, width, _ = img.shape
        img[:, :, index_pop:index_pop + 1] += np.random.normal(
            loc=random_pop[0], scale=random_pop[1],
            size=(height, width, 1)).round()
        img[:, :, index_pop:index_pop + 1] = np.clip(
            img[:, :, index_pop:index_pop + 1],
            self.stats["population"]["min"][0],
            self.stats["population"]["max"][0])

        index_ele = INDECES[DATA_ORDER.index("elevation")]
        img_ele = img[:, :, index_ele:index_ele + 1]
        img_ele_new = np.array(img[:, :, index_ele:index_ele + 1])
        unique_elems = sorted(np.unique(img_ele).tolist())
        rnd = sorted(np.random.normal(
            loc=random_ele[0], scale=random_ele[1], size=len(unique_elems)
        ).round().tolist())
        for i, elem in enumerate(unique_elems):
            img_ele_new[np.where(img_ele == elem)] += rnd[i]
        img[:, :, index_ele:index_ele + 1] = img_ele_new

        index_slo = INDECES[DATA_ORDER.index("slope")]
        img_slo = img[:, :, index_slo:index_slo + 1]
        img_slo_new = np.array(img[:, :, index_slo:index_slo + 1])
        unique_elems = sorted(np.unique(img_slo).tolist())
        rnd = sorted(np.random.normal(
            loc=random_slo[0], scale=random_slo[1], size=len(unique_elems)
        ).round().tolist())
        for i, elem in enumerate(unique_elems):
            img_slo_new[np.where(img_slo == elem)] += rnd[i]
        img[:, :, index_slo:index_slo + 1] = img_slo_new

        index_imgs = INDECES[DATA_ORDER.index("osm_imgs")]
        img[:, :, index_imgs:index_imgs + 3] += np.random.normal(
            loc=random_osm_imgs[0], scale=random_osm_imgs[1],
            size=(height, width, 3)).round()
        img[:, :, index_imgs:index_imgs + 3] = np.clip(
            img[:, :, index_imgs:index_imgs + 3], 0, 255)

        index_roads = INDECES[DATA_ORDER.index("roads")]
        p = np.random.rand(height, width, 1)
        img_roads = img[:, :, index_roads:index_roads + 1]
        img_roads[np.where(np.logical_and(p > random_osm, img_roads > 0))] = 0
        img[:, :, index_roads:index_roads + 1] += img_roads

        index_water = INDECES[DATA_ORDER.index("waterways")]
        p = np.random.rand(height, width, 1)
        img_water = img[:, :, index_water:index_water + 1]
        img_water[np.where(np.logical_and(p > random_osm, img_water > 0))] = 0
        img[:, :, index_water:index_water + 1] += img_water

        index_admin = INDECES[DATA_ORDER.index("admin_bounds")]
        p = np.random.rand(height, width, 1)
        img_admin = img[:, :, index_admin:index_admin + 1]
        img_admin[np.where(np.logical_and(p > random_osm, img_admin > 0))] = 0
        img[:, :, index_admin:index_admin + 1] += img_admin

        return img


class BridgeSampler(Sampler[int]):

    def __init__(self, tile_size, num_samples=None, set_name="train",
                 shuffle=True, train_data=TRAIN_DATA) -> None:
        assert tile_size in [300, 600, 1200], "Tile size not known"
        assert set_name in ["train", "val", "test"], "Set name not known."

        with open(train_data[tile_size]) as f:
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


def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)


def get_dataloaders(
        batch_size: int, tile_size: int, train_data: str = TRAIN_DATA,
        train_metadata: str = TRAIN_METADATA, num_workers: int = 0,
        transform: bool = True, use_augment: bool = True,
        stats_fp: str = STATS_FP) -> Tuple[DataLoader, DataLoader]:
    tr_dataset = BridgeDataset(
        tile_size, use_rnd_pos=True, transform=transform, stats_fp=stats_fp,
        train_data=train_data, train_metadata=train_metadata,
        use_augment=use_augment)
    va_te_dataset = BridgeDataset(
        tile_size, use_rnd_pos=True, transform=transform, stats_fp=stats_fp,
        train_data=train_data, train_metadata=train_metadata,
        use_augment=False)
    sampler_train = BridgeSampler(
        tile_size=tile_size, set_name="train", shuffle=True,
        train_data=train_data)
    sampler_validation = BridgeSampler(
        tile_size=tile_size, set_name="val", shuffle=True,
        train_data=train_data)
    sampler_test = BridgeSampler(
        tile_size=tile_size, set_name="val", shuffle=True,
        train_data=train_data)
    dataloader_train = DataLoader(
        tr_dataset, sampler=sampler_train, batch_size=batch_size,
        worker_init_fn=worker_init_fn, num_workers=num_workers)
    dataloader_validation = DataLoader(
        va_te_dataset, sampler=sampler_validation, batch_size=batch_size,
        worker_init_fn=worker_init_fn, num_workers=num_workers)
    dataloader_test = DataLoader(
        va_te_dataset, sampler=sampler_test, batch_size=batch_size,
        worker_init_fn=worker_init_fn, num_workers=num_workers)
    return dataloader_train, dataloader_validation, dataloader_test
