import geopandas as gpd
import numpy as np
import rasterio
from rasterio.windows import from_bounds
from torch.utils.data import Dataset

from src.scripts.get_value_from_osm_tag import TagMap

DATA_PATHS = {
    "rwanda": {
        "roads": "data/osm/rwanda/roads/rwanda-roads-epsg-32735.tif",
        "waterways": "data/osm/rwanda/waterways/rwanda-waterways-epsg-32735.tif",
        "population": "data/population/rwanda-population-epsg-32735.tif",
        "targets": "data/targets/rwanda-training-data-epsg-32735.shp",
    },
    "uganda": {
        "roads": "data/osm/uganda/roads/uganda-roads-epsg-32735.tif",
        "waterways": "data/osm/uganda/waterways/uganda-waterways-epsg-32735.tif",
        "population": "data/population/uganda-population-epsg-32735.tif",
        "targets": "data/targets/uganda-training-data-epsg-32735.shp",
    },
}
RWANDA_POPULATION_MEAN = 15.153709605481
RWANDA_POPULATION_STD = 17.416015616399
STANDARDIZATION_FUNCTIONS = {
    "roads": lambda road_value: road_value / TagMap.ROADS.value.default,
    "waterways": lambda waterway_value: waterway_value / TagMap.WATERWAYS.value.default,
    "population": lambda population_value: process_population_data(population_value)
}
DESIRED_SHAPE = (200, 200)


def process_population_data(population_data):
    population_data_zero_no_data = np.where(population_data == -99999, -1, population_data)
    population_data_standardized = (population_data_zero_no_data - RWANDA_POPULATION_MEAN) / RWANDA_POPULATION_STD
    return population_data_standardized


class BridgesData(Dataset):
    def __init__(self, tile_size=5000, country="rwanda", inputs=()):
        """
        `inputs` defines desired data to train on, currently accepting:
            - roads (OSM roads data)
            - waterways (OSM waterways data)
            - population (local population data)
        """
        if len(inputs) == 0:
            raise Exception("No input data specified.")
        if country not in list(DATA_PATHS.keys()):
            raise Exception(f"Country {country} is not supported.")
        self.input_files = {_input: DATA_PATHS.get(country).get(_input) for _input in inputs}
        sample_input_file = list(self.input_files.values())[0]
        self.tiles = []
        # TODO: Have statically defined global bounds
        with rasterio.open(sample_input_file) as src:
            bounds = src.bounds
            for tile_lon in np.arange(bounds.left, bounds.right, tile_size):
                for tile_lat in np.arange(bounds.bottom, bounds.top, tile_size):
                    self.tiles.append(
                        (tile_lon, tile_lat, tile_lon + tile_size, tile_lat + tile_size)
                    )

        # Read vector target data into memory
        target_file = DATA_PATHS.get(country).get("targets")
        self.target_data = gpd.read_file(target_file)

    def __len__(self):
        return len(self.tiles)

    # TODO: Preprocess data to have consistent resolution
    def __getitem__(self, item):
        left, bottom, right, top = self.tiles[item]
        rasters = []
        for input_type in self.input_files:
            input_file = self.input_files.get(input_type)
            with rasterio.open(input_file) as src:
                window = from_bounds(left, bottom, right, top, src.transform)
                raster = src.read(
                    1,
                    window=window,
                    out_shape=DESIRED_SHAPE,
                )
                raster_processed = STANDARDIZATION_FUNCTIONS.get(input_type)(raster)
                rasters.append(raster_processed)
        input_channels = np.stack(rasters)
        target_data_in_bounds = self.target_data.cx[left:right, bottom:top]
        target_exists_in_bounds = len(target_data_in_bounds) > 0
        data_item = {"input": input_channels, "target": int(target_exists_in_bounds)}
        return data_item


if __name__ == "__main__":
    data = BridgesData(inputs=("roads", "waterways", "population"))
    print(len(data))
    print(data[0])
