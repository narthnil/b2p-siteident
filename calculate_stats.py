import geopandas as gpd
import json
import rasterio

from rasterio import mask

from src.data import TRAIN_METADATA

STATS = {
    "population": {"mean": [], "std": []},
    "osm_imgs": {"mean": [], "std": []},
    "elevation": {"mean": [], "std": []},
    "slope": {"mean": [], "std": []},
    "roads": {"mean": [], "std": []},
    "waterways": {"mean": [], "std": []},
    "admin_bounds": {"mean": [], "std": []}
}

src = rasterio.open(TRAIN_METADATA["Rwanda"]["population"]["fp"])
flat_pop = src.read(1).flatten()
STATS["population"]["mean"] = [flat_pop[flat_pop != -99999].mean()]
STATS["population"]["std"] = [flat_pop[flat_pop != -99999].std()]

src = rasterio.open(TRAIN_METADATA["Rwanda"]["osm_imgs"]["fp"])
flat_pop = src.read(1).flatten()
STATS["osm_imgs"]["mean"] = [
    src.read(1).mean(), src.read(2).mean(), src.read(3).mean()]
STATS["osm_imgs"]["std"] = [
    src.read(1).std(), src.read(2).std(), src.read(3).std()]

src = rasterio.open(TRAIN_METADATA["Rwanda"]["elevation"]["fp"])
mask_ele = src.read(1).flatten()
flat_ele = src.read(2).flatten()
STATS["elevation"]["mean"] = [flat_ele[mask_ele != 0].mean()]
STATS["elevation"]["std"] = [flat_ele[mask_ele != 0].std()]

src = rasterio.open(TRAIN_METADATA["Rwanda"]["slope"]["fp"])
mask_ele = src.read(1).flatten()
flat_ele = src.read(2).flatten()
STATS["slope"]["mean"] = [flat_ele[mask_ele != 0].mean()]
STATS["slope"]["std"] = [flat_ele[mask_ele != 0].std()]

src = rasterio.open(TRAIN_METADATA["Rwanda"]["roads"]["fp"])
flat_pop = src.read(1).flatten()
STATS["roads"]["mean"] = [src.read(1).mean()]
STATS["roads"]["std"] = [src.read(1).std()]

src = rasterio.open(TRAIN_METADATA["Rwanda"]["waterways"]["fp"])
flat_pop = src.read(1).flatten()
STATS["waterways"]["mean"] = [src.read(1).mean()]
STATS["waterways"]["std"] = [src.read(1).std()]

src = rasterio.open(TRAIN_METADATA["Rwanda"]["admin_bounds"]["fp"])
rwanda_bounds = gpd.read_file(
    "./data/country_masks/used_for_pre_processing/rwanda.shp")
rwanda_admin_bounds = mask.mask(src, rwanda_bounds.geometry, nodata=9999)[0]
flat_rwanda_admin_bounds = rwanda_admin_bounds.flatten()
flat_rwanda_admin_bounds = flat_rwanda_admin_bounds[
    flat_rwanda_admin_bounds != 9999]
flat_rwanda_admin_bounds[flat_rwanda_admin_bounds != 0] = 1
STATS["admin_bounds"]["mean"] = [flat_rwanda_admin_bounds.mean()]
STATS["admin_bounds"]["std"] = [flat_rwanda_admin_bounds.std()]

print(STATS)
