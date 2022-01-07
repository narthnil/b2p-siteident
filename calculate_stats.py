import numpy as np
import json
import rasterio

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

src1 = rasterio.open(TRAIN_METADATA["Rwanda"]["population"]["fp"])
src2 = rasterio.open(TRAIN_METADATA["Uganda"]["population"]["fp"])
flat_pop = np.concatenate((src1.read(1).flatten(), src2.read(1).flatten()))
max_pop = float(np.ceil(flat_pop[flat_pop != -99999].max()))
min_pop = float(np.floor(flat_pop[flat_pop != -99999].min()))
flat_pop[flat_pop != -99999] /= max_pop
print("[Population] max: {} min: {}".format(max_pop, min_pop))
STATS["population"]["mean"] = [float(flat_pop[flat_pop != -99999].mean())]
STATS["population"]["std"] = [float(flat_pop[flat_pop != -99999].std())]
STATS["population"]["max"] = [max_pop]
STATS["population"]["min"] = [min_pop]

src1 = rasterio.open(TRAIN_METADATA["Rwanda"]["osm_imgs"]["fp"])
src2 = rasterio.open(TRAIN_METADATA["Rwanda"]["osm_imgs"]["fp"])
flat_c1 = np.concatenate((
    src1.read(1).flatten(), src2.read(1).flatten())) / 255
flat_c2 = np.concatenate((
    src1.read(2).flatten(), src2.read(2).flatten())) / 255
flat_c3 = np.concatenate((
    src1.read(3).flatten(), src2.read(3).flatten())) / 255
print("[OSM image] max: ({}, {}, {}) min: ({}, {}, {})".format(
    flat_c1.max(), flat_c2.max(), flat_c3.max(),
    flat_c1.min(), flat_c2.min(), flat_c3.min()
))
STATS["osm_imgs"]["mean"] = [
    float(flat_c1.mean()), float(flat_c2.mean()), float(flat_c3.mean())]
STATS["osm_imgs"]["std"] = [
    float(flat_c1.std()), float(flat_c2.std()), float(flat_c3.std())]
STATS["osm_imgs"]["max"] = [255, 255, 255]
STATS["osm_imgs"]["min"] = [0, 0, 0]

src1 = rasterio.open(TRAIN_METADATA["Rwanda"]["elevation"]["fp"])
src2 = rasterio.open(TRAIN_METADATA["Uganda"]["elevation"]["fp"])

mask_ele1 = src1.read(1).flatten()
flat_ele1 = src1.read(2).flatten()
mask_ele2 = src2.read(1).flatten()
flat_ele2 = src2.read(2).flatten()

flat_ele = np.concatenate((
    flat_ele1[mask_ele1 != 0], flat_ele2[mask_ele2 != 0])).astype(np.float64)
max_ele = float(flat_ele.max())
min_ele = float(flat_ele.min())
flat_ele /= 255
print("[Elevation] max: {} min: {}".format(max_ele, min_ele))
STATS["elevation"]["mean"] = [float(flat_ele.mean())]
STATS["elevation"]["std"] = [float(flat_ele.std())]
STATS["elevation"]["max"] = [255.]
STATS["elevation"]["min"] = [0.]

src1 = rasterio.open(TRAIN_METADATA["Rwanda"]["slope"]["fp"])
src2 = rasterio.open(TRAIN_METADATA["Uganda"]["slope"]["fp"])
mask_slo1 = src1.read(1).flatten()
flat_slo1 = src1.read(2).flatten()
mask_slo2 = src2.read(1).flatten()
flat_slo2 = src2.read(2).flatten()
flat_slo = np.concatenate((
    flat_slo1[mask_slo1 != 0], flat_slo2[mask_slo2 != 0])).astype(np.float64)
max_slo = float(flat_slo.max())
min_slo = float(flat_slo.min())
flat_slo /= 255.
print("[Slope] max: {} min: {}".format(max_slo, min_slo))
STATS["slope"]["mean"] = [float(flat_slo.mean())]
STATS["slope"]["std"] = [float(flat_slo.std())]
STATS["slope"]["max"] = [255.]
STATS["slope"]["min"] = [0.]

src1 = rasterio.open(TRAIN_METADATA["Rwanda"]["roads"]["fp"])
src2 = rasterio.open(TRAIN_METADATA["Uganda"]["roads"]["fp"])
flat_roads = np.concatenate((src1.read(1).flatten(), src2.read(1).flatten()))
max_roads = float(flat_roads.max())
min_roads = float(flat_roads.min())
flat_roads /= max_roads
print("[Roads] max: {} min: {}".format(max_roads, min_roads))
STATS["roads"]["mean"] = [float(flat_roads.mean())]
STATS["roads"]["std"] = [float(flat_roads.std())]
STATS["roads"]["max"] = [max_roads]
STATS["roads"]["min"] = [min_roads]

src1 = rasterio.open(TRAIN_METADATA["Rwanda"]["waterways"]["fp"])
src2 = rasterio.open(TRAIN_METADATA["Uganda"]["waterways"]["fp"])
flat_water = np.concatenate((src1.read(1).flatten(), src2.read(1).flatten()))
max_water = float(flat_water.max())
min_water = float(flat_water.min())
flat_water /= max_water
print("[Water] max: {} min: {}".format(max_water, min_water))
STATS["waterways"]["mean"] = [float(flat_water.mean())]
STATS["waterways"]["std"] = [float(flat_water.std())]
STATS["waterways"]["max"] = [max_water]
STATS["waterways"]["min"] = [min_water]


src1 = rasterio.open(TRAIN_METADATA["Rwanda"]["admin_bounds"]["fp"])
src2 = rasterio.open(TRAIN_METADATA["Uganda"]["admin_bounds"]["fp"])
flat_admin = np.concatenate((src1.read(1).flatten(), src2.read(1).flatten()))
flat_admin = flat_admin[flat_admin != 0.]
flat_admin[flat_admin == 127.] = 0
max_admin = float(np.ceil(flat_admin.max()))
min_admin = float(np.floor(flat_admin.min()))
print("[Admin] max: {} min: {}".format(max_admin, min_admin))
STATS["admin_bounds"]["mean"] = [float(flat_admin.mean())]
STATS["admin_bounds"]["std"] = [float(flat_admin.std())]
STATS["admin_bounds"]["max"] = [1.]
STATS["admin_bounds"]["min"] = [0.]

with open("data/ground_truth/stats.json", "w+") as f:
    json.dump(STATS, f, indent=4)
