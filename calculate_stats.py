import numpy as np
import json
import rasterio

from src.data import METADATA

if __name__ == "__main__":
    STATS = {
        k: {} for k in [
            "population", "osm_img", "elevation", "slope", "roads",
            "waterways", "admin_bounds_qgis", "admin_bounds_gadm"]}
    # population data

    # load rwanda data
    src1 = rasterio.open(METADATA["Rwanda"]["population"]["fp"])
    # load uganda data
    src2 = rasterio.open(METADATA["Uganda"]["population"]["fp"])
    # flatten both images and concatenate
    flat_pop = np.concatenate((src1.read(1).flatten(), src2.read(1).flatten()))
    # get max population values, ignore -99999 (invalid data value)
    max_pop = float(np.ceil(flat_pop[flat_pop != -99999].max()))
    # get min population values, ignore -99999 (invalid data value)
    min_pop = float(np.floor(flat_pop[flat_pop != -99999].min()))
    # divide all valid population values by max_pop, essentially scaling the
    # image to values between 0 and 1
    flat_pop[flat_pop != -99999] /= max_pop
    print("[Population] max: {} min: {}".format(max_pop, min_pop))
    # calculate mean and std.
    STATS["population"]["mean"] = [float(flat_pop[flat_pop != -99999].mean())]
    STATS["population"]["std"] = [float(flat_pop[flat_pop != -99999].std())]
    STATS["population"]["max"] = [max_pop]
    STATS["population"]["min"] = [min_pop]

    # opening rwanda osm image
    src1 = rasterio.open(METADATA["Rwanda"]["osm_img"]["fp"])
    # opening uganda osm image
    src2 = rasterio.open(METADATA["Rwanda"]["osm_img"]["fp"])
    # concatenate first channel of rwanda and uganda, divide all values by max
    # value (=255)
    flat_c1 = np.concatenate((
        src1.read(1).flatten(), src2.read(1).flatten())) / 255
    # concatenate second channel of rwanda and uganda, divide all values by max
    # value (=255)
    flat_c2 = np.concatenate((
        src1.read(2).flatten(), src2.read(2).flatten())) / 255
    # concatenate third channel of rwanda and uganda, divide all values by max
    # value (=255)
    flat_c3 = np.concatenate((
        src1.read(3).flatten(), src2.read(3).flatten())) / 255
    print("[OSM image] max: ({}, {}, {}) min: ({:.2f}, {:.2f}, {:.2f})".format(
        flat_c1.max(), flat_c2.max(), flat_c3.max(),
        flat_c1.min(), flat_c2.min(), flat_c3.min()
    ))
    # calculate mean and std.
    STATS["osm_img"]["mean"] = [
        float(flat_c1.mean()), float(flat_c2.mean()), float(flat_c3.mean())]
    STATS["osm_img"]["std"] = [
        float(flat_c1.std()), float(flat_c2.std()), float(flat_c3.std())]
    STATS["osm_img"]["max"] = [255, 255, 255]
    STATS["osm_img"]["min"] = [0, 0, 0]

    # opening rwanda elevation data
    src1 = rasterio.open(METADATA["Rwanda"]["elevation"]["fp"])
    # opening uganda data
    src2 = rasterio.open(METADATA["Uganda"]["elevation"]["fp"])

    # read two channels, first channel denotes valid/invalid pixels, second
    # channel denotes the actual elevation values
    # flatten all images
    mask_ele1 = src1.read(1).flatten()
    flat_ele1 = src1.read(2).flatten()
    mask_ele2 = src2.read(1).flatten()
    flat_ele2 = src2.read(2).flatten()

    # mask out invalid pixels (=0), concatenate uganda and rwandas valid pixels
    flat_ele = np.concatenate((
        flat_ele1[mask_ele1 != 0], flat_ele2[mask_ele2 != 0])).astype(
            np.float64)
    # calculate max and min elevation values
    max_ele = float(flat_ele.max())
    min_ele = float(flat_ele.min())
    # max elevation = 255, divide by max elevation to scale elevation to [0, 1]
    flat_ele /= 255
    print("[Elevation] max: {} min: {}".format(max_ele, min_ele))
    # calculate mean and std
    STATS["elevation"]["mean"] = [float(flat_ele.mean())]
    STATS["elevation"]["std"] = [float(flat_ele.std())]
    STATS["elevation"]["max"] = [255.]
    STATS["elevation"]["min"] = [0.]

    # opening rwanda slope
    src1 = rasterio.open(METADATA["Rwanda"]["slope"]["fp"])
    # opening uganda slope
    src2 = rasterio.open(METADATA["Uganda"]["slope"]["fp"])
    # read two channels, first channel denotes valid/invalid pixels, second
    # channel denotes the actual slope values
    # flatten all images
    mask_slo1 = src1.read(1).flatten()
    flat_slo1 = src1.read(2).flatten()
    mask_slo2 = src2.read(1).flatten()
    flat_slo2 = src2.read(2).flatten()
    flat_slo = np.concatenate((
        flat_slo1[mask_slo1 != 0], flat_slo2[mask_slo2 != 0])).astype(
            np.float64)
    # calculate max and min slope values
    max_slo = float(flat_slo.max())
    min_slo = float(flat_slo.min())
    # max slope = 255, divide by max slope to scale elevation to [0, 1]
    flat_slo /= 255.
    print("[Slope] max: {} min: {}".format(max_slo, min_slo))
    # calculate mean and std
    STATS["slope"]["mean"] = [float(flat_slo.mean())]
    STATS["slope"]["std"] = [float(flat_slo.std())]
    STATS["slope"]["max"] = [255.]
    STATS["slope"]["min"] = [0.]

    # opening rwanda roads (from osm)
    src1 = rasterio.open(METADATA["Rwanda"]["roads"]["fp"])
    # opening uganda roads (from osm)
    src2 = rasterio.open(METADATA["Uganda"]["roads"]["fp"])
    # flatten both images, and concatenate uganda and rwanda images
    flat_roads = np.concatenate(
        (src1.read(1).flatten(), src2.read(1).flatten()))
    # calculate min and max values (it should be 0-7, see
    # src.scripts.get_value_from_osm_tag)
    max_roads = float(flat_roads.max())
    min_roads = float(flat_roads.min())
    # divide by max road value
    flat_roads /= max_roads
    print("[Roads] max: {} min: {}".format(max_roads, min_roads))
    # calculate mean and std
    STATS["roads"]["mean"] = [float(flat_roads.mean())]
    STATS["roads"]["std"] = [float(flat_roads.std())]
    STATS["roads"]["max"] = [max_roads]
    STATS["roads"]["min"] = [min_roads]

    # opening rwanda waterways (from osm)
    src1 = rasterio.open(METADATA["Rwanda"]["waterways"]["fp"])
    # opening uganda waterways (from osm)
    src2 = rasterio.open(METADATA["Uganda"]["waterways"]["fp"])
    # flatten both images, and concatenate uganda and rwanda images
    flat_water = np.concatenate(
        (src1.read(1).flatten(), src2.read(1).flatten()))
    # calculate min and max values (it should be 0-3, see
    # src.scripts.get_value_from_osm_tag)
    max_water = float(flat_water.max())
    min_water = float(flat_water.min())
    # divide by max waterways value
    flat_water /= max_water
    print("[Water] max: {} min: {}".format(max_water, min_water))
    # calculate mean and std
    STATS["waterways"]["mean"] = [float(flat_water.mean())]
    STATS["waterways"]["std"] = [float(flat_water.std())]
    STATS["waterways"]["max"] = [max_water]
    STATS["waterways"]["min"] = [min_water]

    # read rwanda admin boundaries extracted from qgis
    src1 = rasterio.open(METADATA["Rwanda"]["admin_bounds_qgis"]["fp"])
    # read uganda admin boundaries extracted from qgis
    src2 = rasterio.open(METADATA["Uganda"]["admin_bounds_qgis"]["fp"])
    # flatten both images, and concatenate uganda and rwanda images
    flat_admin = np.concatenate(
        (src1.read(1).flatten(), src2.read(1).flatten()))
    # calculate min and max values (should be 0, 1)
    max_admin = float(np.ceil(flat_admin.max()))
    min_admin = float(np.floor(flat_admin.min()))
    print("[Admin (QGIS)] max: {} min: {}".format(max_admin, min_admin))
    # calculate mean and std
    STATS["admin_bounds_qgis"]["mean"] = [float(flat_admin.mean())]
    STATS["admin_bounds_qgis"]["std"] = [float(flat_admin.std())]
    STATS["admin_bounds_qgis"]["max"] = [1.]
    STATS["admin_bounds_qgis"]["min"] = [0.]

    # read rwanda admin boundaries extracted from qgis
    src1 = rasterio.open(METADATA["Rwanda"]["admin_bounds_gadm"]["fp"])
    # read uganda admin boundaries extracted from qgis
    src2 = rasterio.open(METADATA["Uganda"]["admin_bounds_gadm"]["fp"])
    # flatten both images, and concatenate uganda and rwanda images
    flat_admin = np.concatenate(
        (src1.read(1).flatten(), src2.read(1).flatten()))
    flat_admin[flat_admin >= 1.] = 1.
    # calculate min and max values (should be 0, 1)
    max_admin = float(np.ceil(flat_admin.max()))
    min_admin = float(np.floor(flat_admin.min()))
    print("[Admin (GADM)] max: {} min: {}".format(max_admin, min_admin))
    # calculate mean and std
    STATS["admin_bounds_gadm"]["mean"] = [float(flat_admin.mean())]
    STATS["admin_bounds_gadm"]["std"] = [float(flat_admin.std())]
    STATS["admin_bounds_gadm"]["max"] = [1.]
    STATS["admin_bounds_gadm"]["min"] = [0.]

    # save to json file
    # with open("data/ground_truth/stats.json", "w+") as f:
    #     json.dump(STATS, f, indent=4)
