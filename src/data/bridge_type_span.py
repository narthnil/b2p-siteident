import itertools
import json

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import rasterio

import torch

from torchvision import transforms
from torch.utils.data import DataLoader, WeightedRandomSampler

from src import utils
from src.data import transforms as data_transf
from src.data.bridge_site import (
    BridgeDataset, DATA_ORDER, METADATA, STATS_FP, TRAIN_DATA)
from src.data.geometry import get_tile_bounds, shift_coords


LABEL_NAMES_2_VALUES = {
    "feasibility": {
        "Not feasible": 0,
        "Feasible": 1
    },
    "type": {
        "Suspension": 0,
        "Suspended": 1,
        "Short Span (11-30m)": 2,
        "Culvert (10m or less)": 3,
        "Other": 4,
        "N/A": -1,
    }
}

LABEL_VALUES_2_NAMES = {
    lab: {val: name for val, name in LABEL_NAMES_2_VALUES[lab].items()}
    for lab in LABEL_NAMES_2_VALUES}

LABEL_NAMES_2_COL_NAMES = {
    "feasibility": "Bridge Feasibility",
    "type": "Bridge Type",
    "span": "Span"
}

LABEL_NAMES_2_DATA_TYPE = {
    "feasibility": str,
    "type": str,
    "span": float
}

V1_FPATH = "data/bridge_type_span_data/data_v1.csv"
V2_FPATH = "data/bridge_type_span_data/data_v2.csv"


class BridgeTypeSpanDataset(BridgeDataset):
    """Dataset module to load all TIF-based data from file and extract tiles.
    """

    def __init__(self, data_fp: str = "data/bridge_type_span_data/data_v1.csv",
                 data_order: List[str] = DATA_ORDER,
                 raster_data: Dict = METADATA, stats_fp: str = STATS_FP,
                 tile_size: int = 300, transform: bool = True,
                 use_augment: bool = True,
                 use_rnd_center_point: bool = True,
                 set_name: str = "train",
                 labels: List = ["feasibility", "type", "span"]) -> None:
        """
        Args:
            data_fp (str, optional): CSV file path to the data.
            data_order (str, optional): The order of the data modalities to be
                loaded and read. Default: DATA_ORDER.
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
            set_name (str, optional): Name of the subset to be used. Can only 
                be either `train`, `val` or `test`.
        """
        assert tile_size in [300, 600, 1200], "Tile size not known."
        assert set_name in ["train", "val", "test"], "Set name not known"
        assert all([l in ["feasibility", "type", "span"] for l in labels]), \
            "At least one label not known."
        assert len(labels) <= 3, \
            "Expected at most 3 different types of labels."

        self.data_order = data_order
        self.tile_size = tile_size
        self.use_rnd_center_point = use_rnd_center_point
        self.transform = transform
        self.use_augment = use_augment
        self.train_metadata = raster_data
        self.set_name = set_name
        self.labels = sorted(labels)

        # read csv train data
        self.data = pd.read_csv(data_fp)
        self.data = self.data[self.data.split.str.startswith(set_name)]
        # drop missing values
        if "feasibility" in labels:
            if "type" in labels:
                self.data = self.data.drop(
                    self.data[(self.data["Bridge Feasibility"] == "Feasible")
                              & self.data["Bridge Type"].isna()].index)
            if "span" in labels:
                self.data = self.data.drop(
                    self.data[(self.data["Bridge Feasibility"] == "Feasible")
                              & self.data["Span"].isna()].index)
        else:
            if "type" in labels:
                self.data = self.data.drop(
                    self.data[self.data["Bridge Type"].isna()].index)
            if "span" in labels:
                self.data = self.data.drop(
                    self.data[self.data["Span"].isna()].index)
        self.data["Bridge Type"].fillna(value="N/A", inplace=True)
        self.data["Span"].fillna(value=-1, inplace=True)
        self.data.reset_index(drop=True, inplace=True)
        # sort values
        self.data.sort_values(by="Opportunity Unique Identifier", inplace=True)
        self.data.reset_index(drop=True, inplace=True)

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
            data_transf.Normalize(
                list(itertools.chain(
                    *[self.stats[name]["mean"] for name in self.data_order])),
                list(itertools.chain(
                    *[self.stats[name]["std"] for name in self.data_order])))

        ])
        # undo normalization function
        self.invert_transform_func = transforms.Compose([
            data_transf.UnNormalize(
                list(itertools.chain(
                    *[self.stats[name]["mean"] for name in self.data_order])),
                list(itertools.chain(
                    *[self.stats[name]["std"] for name in self.data_order])))

        ])

    def get_label_values(self, idx: int) -> List:
        entry = self.data.iloc[idx]
        label_names = [
            entry[LABEL_NAMES_2_COL_NAMES[lab_name]]
            for lab_name in self.labels]
        label_values = []
        for lab, lab_name in zip(label_names, self.labels):
            if lab_name == "span":
                label_values.append(lab)
            else:
                label_values.append(LABEL_NAMES_2_VALUES[lab_name][lab])
        return label_values

    def convert_label_values_2_names(self, labels) -> Dict:
        labels_dict = {}
        for lab, lab_name in zip(labels, self.labels):
            if lab_name == "span":
                labels_dict[lab_name] = lab
            else:
                labels_dict[lab_name] = LABEL_VALUES_2_NAMES[lab_name][lab]
        return labels_dict

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor]:
        """Get images and label based on index"""
        # get dataset entry
        entry = self.data.iloc[idx]
        country = entry.Country
        lon = entry["Bridge Opportunity: GPS (Longitude)"]
        lat = entry["Bridge Opportunity: GPS (Latitude)"]
        lonlats = []
        if self.use_rnd_center_point:
            for _ in range(10):  # have more points for backup
                lon_shift, lat_shift = np.random.normal(
                    loc=0.0, scale=5.0, size=2).tolist()
                lonlats.append(shift_coords(lon, lat, lon_shift, lat_shift))
        lonlats.append((lon, lat))

        images = []
        for lon, lat in lonlats:
            # get bounds
            left, bottom, right, top = get_tile_bounds(
                lon, lat, self.tile_size)
            # get images
            try:
                imgs = self.get_imgs(left, bottom, right, top, entry.Country)
            except Exception as e:
                print("[Warning]", e)
                print("[Warning]", lonlats[-1][0], lonlats[-1][1])
                print("[Warning]", idx, left, bottom,
                      right, top, entry.Country)
                print("[Warning] Use back-up sampled points.")
                continue
            # augment
            if self.use_augment:
                imgs = self.augment(imgs)
            # create torch.Tensor and permute dimensions from (width, height,
            # channels) to channels, width, height
            imgs = torch.from_numpy(imgs.copy()).permute(2, 0, 1)
            # normalizes
            if self.transform:
                imgs = self.transform_imgs(imgs)
            images.append(imgs)
            if len(images) == 1:
                break
        labels = self.get_label_values(idx)
        return images[0], labels

    def __len__(self) -> int:
        """Returns length of the dataset"""
        return len(self.data)


def get_dataloaders(batch_size: int, tile_size: int,
                    data_version: str = "v1",
                    balanced_sampling: bool = False,
                    data_order: List = DATA_ORDER,
                    labels: List = ["feasibility", "type", "span"],
                    num_test_samples: int = 64, num_workers: int = 0,
                    stats_fp: str = STATS_FP, test_batch_size: int = 10,
                    transform: bool = True, train_data: str = TRAIN_DATA,
                    use_augment: bool = True,
                    use_rnd_center_point: bool = True,
                    use_several_test_samples: bool = False,
                    num_samples: int = 2000) -> Tuple[
        DataLoader, DataLoader]:
    """Returns dataloaders for training and evaluation.

        Args:
            batch_size (int): Batch size of dataloaders.
            balanced_sampling (bool, optional): Whether use balanced sampling 
                or not.
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
    assert data_version in ["v1", "v2"], "Data version not known"
    assert all([l in ["feasibility", "type", "span"] for l in labels]), \
        "At least one label not known."
    assert len(labels) <= 3, "Expected at most 3 different types of labels."
    fpath = V1_FPATH if data_version == "v1" else V2_FPATH

    tr_dataset = BridgeTypeSpanDataset(
        fpath, data_order=data_order, num_test_samples=num_test_samples,
        stats_fp=stats_fp, tile_size=tile_size, transform=transform,
        use_augment=use_augment, use_rnd_center_point=use_rnd_center_point,
        set_name="train", labels=labels)
    va_dataset = BridgeTypeSpanDataset(
        fpath, data_order=data_order, num_test_samples=num_test_samples,
        stats_fp=stats_fp, tile_size=tile_size, transform=transform,
        use_augment=use_several_test_samples and use_augment,
        use_rnd_center_point=use_several_test_samples and use_rnd_center_point,
        set_name="val", labels=labels)
    te_dataset = BridgeTypeSpanDataset(
        fpath, data_order=data_order, num_test_samples=num_test_samples,
        stats_fp=stats_fp, tile_size=tile_size, transform=transform,
        use_augment=use_several_test_samples and use_augment,
        use_rnd_center_point=use_several_test_samples and use_rnd_center_point,
        set_name="test", labels=labels)

    if balanced_sampling is True:
        sampler = WeightedRandomSampler(
            num_samples=num_samples, replacement=True)
    else:
        sampler = None

    # dataloaders
    common_loader_kwargs = {
        "worker_init_fn": utils.worker_init_fn,
        "num_workers": num_workers
    }
    te_batch_size = test_batch_size if use_several_test_samples else batch_size
    dataloader_train = DataLoader(
        tr_dataset, sampler=sampler, batch_size=batch_size,
        drop_last=True, **common_loader_kwargs)
    dataloader_validation = DataLoader(
        va_dataset, batch_size=te_batch_size, **common_loader_kwargs,
        shuffle=False)
    dataloader_test = DataLoader(
        te_dataset, batch_size=te_batch_size, **common_loader_kwargs,
        shuffle=False)

    # unlabelled dataset
    uganda_dataset = NoLabelTileDataset(
        data=train_data, data_order=data_order, data_version=data_version,
        len_dataset=len(sampler_train) * 2, raster_data=train_metadata,
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
