"""
Data V1 splits Rwanda into 11 parts equally by its longitude. Parts 1-5 and
7-11 are used for training, part 6 is used for validation. Similarly, Uganda is 
split into 11 parts equally by its longitude. Parts 1-9 are used for testingand
10 are used for training, part 6 is used for validation.

For the semi-supervised learning (ssl) version, we use the Uganda data without
any labels.
"""
import geopandas as gpd
import pandas as pd

from shapely.geometry import Polygon

from preprocess_train_data_v1 import (
    get_dfs, get_bounds, get_rwanda_va_range, get_polygon_gdf_from_bounds,
    intersect, diff, set_split_and_min_max_vals, format_neg_entries)
from preprocess_train_data_v1 import CRS, THRES
from src.data import geometry


def get_uganda_tr_range(bounds):
    """Getting the longitude range for Uganda train set.
    We are splitting Uganda into 11 parts. The night and tenth parts are used
    for train, the last part is used for validation.

    Args:
        bounds (Dict): Contains the bounds for Uganda.

    Returns:
        val_range (Tuple): The tuple consists of two latitude values
            representing the minimum and maximum latitude values that the
            validation data can have.
    """
    tr_range = (
        bounds["uganda"]["left"] + (
            bounds["uganda"]["right"] - bounds["uganda"]["left"]) / 11 * 7,
        bounds["uganda"]["left"] + (
            bounds["uganda"]["right"] - bounds["uganda"]["left"]) / 11 * 9
    )
    return tr_range


if __name__ == "__main__":
    lat_name = "GPS (Latitude)"
    lon_name = "GPS (Longitude)"

    # get data bounds for Rwanda and Uganda
    bounds = get_bounds()
    # get validation range for Rwanda
    va_range = get_rwanda_va_range(bounds)
    # get test range for Uganda
    tr_range = get_uganda_tr_range(bounds)

    # loading shape files containing country bounds
    rwanda_bounds = gpd.read_file("./data/country_masks/rwanda.shp")
    uganda_bounds = gpd.read_file("./data/country_masks/uganda.shp")

    # intersect the polygon with the testing bounds (left of the training
    # bounds) with the official Rwanda country bounds
    ug_te_bounds = {
        "left": bounds["uganda"]["left"], "right": tr_range[0],
        "top": bounds["uganda"]["top"], "bottom": bounds["uganda"]["bottom"]}
    ug_te_bounds = get_polygon_gdf_from_bounds(ug_te_bounds)
    uganda_te_bounds = intersect(ug_te_bounds, uganda_bounds)

    # intersect the polygon with the training bounds with the official Rwanda
    # country bounds
    ug_tr_bounds = {
        "left": tr_range[0], "right": tr_range[1],
        "top": bounds["uganda"]["top"], "bottom": bounds["uganda"]["bottom"]}
    ug_tr_bounds = get_polygon_gdf_from_bounds(ug_tr_bounds)
    uganda_tr_bounds = intersect(ug_tr_bounds, uganda_bounds)

    # intersect the polygon with the validation bounds (right of the training
    # bounds) with the official Uganda country bounds
    ug_va_bounds = {
        "left": tr_range[1], "right": bounds["uganda"]["right"],
        "top": bounds["uganda"]["top"], "bottom": bounds["uganda"]["bottom"]}
    ug_va_bounds = get_polygon_gdf_from_bounds(ug_va_bounds)
    uganda_va_bounds = intersect(ug_va_bounds, uganda_bounds)

    # intersect the polygon with the training bounds (left of the validation
    # bounds) with the official Rwanda country bounds
    tr_lower_bounds = {
        "left": bounds["rwanda"]["left"], "right": va_range[0],
        "top": bounds["rwanda"]["top"], "bottom": bounds["rwanda"]["bottom"]}
    rwanda_tr_lower_bounds = get_polygon_gdf_from_bounds(tr_lower_bounds)
    rwanda_tr_lower_bounds = intersect(rwanda_tr_lower_bounds, rwanda_bounds)

    # intersect the polygon with the validation bounds with the official Rwanda
    # country bounds
    va_bounds = {
        "left": va_range[0], "right": va_range[1],
        "top": bounds["rwanda"]["top"], "bottom": bounds["rwanda"]["bottom"]}
    rwanda_va_bounds = get_polygon_gdf_from_bounds(va_bounds)
    rwanda_va_bounds = intersect(rwanda_va_bounds, rwanda_bounds)

    # intersect the polygon with the training bounds (right of the validation
    # bounds) with the official Rwanda country bounds
    tr_upper_bounds = {
        "left": va_range[1], "right": bounds["rwanda"]["right"],
        "top": bounds["rwanda"]["top"], "bottom": bounds["rwanda"]["bottom"]}
    rwanda_tr_upper_bounds = get_polygon_gdf_from_bounds(tr_upper_bounds)
    rwanda_tr_upper_bounds = intersect(rwanda_tr_upper_bounds, rwanda_bounds)

    for tile_size in [300, 600, 1200]:
        rwanda_df, uganda_df = get_dfs()

        pos_size = tile_size - THRES
        non_neg_size = 2 * tile_size - THRES
        # rwanda
        recid_name = "Record ID"
        # adding prefix `rw-` to every Rwanda bridge entry to make sure it's
        # unique when combining with Uganda data
        rwanda_df[recid_name] = "rw-" + rwanda_df[recid_name]
        rwanda_df = rwanda_df.rename(columns={recid_name: "id"})
        # drop `Site Name` column
        rwanda_df.drop(["Site Name"], inplace=True, axis=1)
        # defines the area where bridge sites can lie in
        rwanda_df["pos_polygon"] = rwanda_df.apply(
            lambda x: Polygon(geometry.get_square_area(
                x[lon_name], x[lat_name], square_length=pos_size)), axis=1)
        # non negative area where cannot be a `negative` bridge site point
        rwanda_df["non_neg_polygon"] = rwanda_df.apply(
            lambda x: Polygon(
                geometry.get_square_area(
                    x[lon_name], x[lat_name], square_length=non_neg_size)),
            axis=1)

        # separate rows in Rwanda bridge data into `train_lower`, `val`,
        # `train_upper` according to bounds
        set_split_and_min_max_vals(rwanda_df, "train_lower", lon_name,
                                   bounds["rwanda"]["left"], va_range[0])
        set_split_and_min_max_vals(rwanda_df, "val", lon_name,
                                   va_range[0], va_range[1])
        set_split_and_min_max_vals(rwanda_df, "train_upper", lon_name,
                                   va_range[1], bounds["rwanda"]["right"])

        # adding prefix `ug-` to every Uganda bridge entry to make sure it's
        # unique when combining with Rwanda data
        recid_name = "Opportunity Unique Identifier"
        uganda_df[recid_name] = "ug-" + uganda_df[recid_name]
        uganda_df = uganda_df.rename(columns={recid_name: "id"})
        uganda_df.drop(["Opportunity ID"], inplace=True, axis=1)

        # defines the area where bridge sites can lie in
        uganda_df["pos_polygon"] = uganda_df.apply(
            lambda x: Polygon(geometry.get_square_area(
                x[lon_name], x[lat_name], square_length=pos_size)), axis=1)
        # non negative area where cannot be a `negative` bridge site point
        uganda_df["non_neg_polygon"] = uganda_df.apply(
            lambda x: Polygon(geometry.get_square_area(
                x[lon_name], x[lat_name], square_length=non_neg_size)), axis=1)

        # separate rows in Uganda bridge data into `test`, `train`,
        # `val` according to bounds
        set_split_and_min_max_vals(uganda_df, "test", lon_name,
                                   bounds["uganda"]["left"], tr_range[0])
        set_split_and_min_max_vals(uganda_df, "train", lon_name,
                                   tr_range[0], tr_range[1])
        set_split_and_min_max_vals(uganda_df, "val", lon_name,
                                   tr_range[1], bounds["uganda"]["right"])

        # for each set of Rwanda (train_lower, val, train_upper), drop
        # pos_polygon and rename non_negative_polygon to geometry
        nn_rwanda_tr_lower = gpd.GeoDataFrame(
            rwanda_df[rwanda_df.split == "train_lower"].drop(
                "pos_polygon", axis=1).rename(
                    {"non_neg_polygon": "geometry"}, axis=1), crs=CRS)
        nn_rwanda_va = gpd.GeoDataFrame(
            rwanda_df[rwanda_df.split == "val"].drop(
                "pos_polygon", axis=1).rename(
                    {"non_neg_polygon": "geometry"}, axis=1), crs=CRS)
        nn_rwanda_tr_upper = gpd.GeoDataFrame(
            rwanda_df[rwanda_df.split == "train_upper"].drop(
                "pos_polygon", axis=1).rename(
                    {"non_neg_polygon": "geometry"}, axis=1), crs=CRS)

        # for each set of Uganda (test, val, test), drop
        # pos_polygon and rename non_negative_polygon to geometry
        nn_uganda_te = gpd.GeoDataFrame(
            uganda_df[uganda_df.split == "test"].drop(
                "pos_polygon", axis=1).rename(
                    {"non_neg_polygon": "geometry"}, axis=1), crs=CRS)
        nn_uganda_tr = gpd.GeoDataFrame(
            uganda_df[uganda_df.split == "train"].drop(
                "pos_polygon", axis=1).rename(
                    {"non_neg_polygon": "geometry"}, axis=1), crs=CRS)
        nn_uganda_va = gpd.GeoDataFrame(
            uganda_df[uganda_df.split == "val"].drop(
                "pos_polygon", axis=1).rename(
                    {"non_neg_polygon": "geometry"}, axis=1), crs=CRS)

        # the area in which center points for `negative` tiles is obtained by
        # by calculating the geometries that are part of the bounds but are not
        # contained in non negative geometry (== difference)
        neg_rwanda_tr_lower = diff(rwanda_tr_lower_bounds, nn_rwanda_tr_lower)
        neg_rwanda_va = diff(rwanda_va_bounds, nn_rwanda_va)
        neg_rwanda_tr_upper = diff(rwanda_tr_upper_bounds, nn_rwanda_tr_upper)

        neg_uganda_te = diff(uganda_te_bounds, nn_uganda_te)
        neg_uganda_tr = diff(uganda_tr_bounds, nn_uganda_tr)
        neg_uganda_va = diff(uganda_va_bounds, nn_uganda_va)

        rwanda_df["Country"] = "Rwanda"
        uganda_df["Country"] = "Uganda"

        # drop non-negative polygons, rename positive polygon (pos_polygon) to
        # geometry
        # sampled pos_polygon represents the area where tile center points can
        # be
        df = pd.concat([rwanda_df, uganda_df]).drop(
            "non_neg_polygon", axis=1).rename({"pos_polygon": "geometry"},
                                              axis=1)
        # create pos_neg column to denote positive and negative entries
        df["pos_neg"] = "pos"

        # format negative entries to have unique indeces, min_x & max_x as
        # bounds as well as naming the split and country
        neg_rwanda_tr_lower = format_neg_entries(
            neg_rwanda_tr_lower, "rw-neg-tr-lower-", bounds["rwanda"]["left"],
            va_range[0], "train-lower", "Rwanda")
        neg_rwanda_va = format_neg_entries(
            neg_rwanda_va, "rw-neg-val-", va_range[0], va_range[1], "val",
            "Rwanda")
        neg_rwanda_tr_upper = format_neg_entries(
            neg_rwanda_tr_upper, "rw-neg-tr-upper-", va_range[1],
            bounds["rwanda"]["right"], "train-upper", "Rwanda")

        neg_uganda_te = format_neg_entries(
            neg_uganda_te, "ug-neg-te-", bounds["uganda"]["left"],
            tr_range[0], "test", "Uganda")
        neg_uganda_tr = format_neg_entries(
            neg_uganda_tr, "ug-neg-tr-", tr_range[0], tr_range[1],
            "train", "Uganda")
        neg_uganda_va = format_neg_entries(
            neg_uganda_va, "ug-neg-va-", tr_range[1],
            bounds["uganda"]["right"], "val", "Uganda")

        # concatenate all negative entries
        negs = pd.concat([neg_rwanda_tr_lower, neg_rwanda_tr_upper,
                          neg_rwanda_va, neg_uganda_te,
                          neg_uganda_va, neg_uganda_tr])
        negs["pos_neg"] = "neg"
        # concatenate positive and negative entries
        df_pos_neg = gpd.GeoDataFrame(
            pd.concat([df, negs]).reset_index(drop=True), crs=CRS)
        # save to file
        df_pos_neg.to_file(
            "./data/ground_truth/train_{}_v2.geojson".format(tile_size))

    # adding indeces and splits to all country and train/val/test bounds
    rwanda_tr_lower_bounds["id"] = "rw-tr-lo-0"
    rwanda_tr_lower_bounds["split"] = "train"

    rwanda_va_bounds["id"] = "rw-va-0"
    rwanda_va_bounds["split"] = "val"

    rwanda_tr_upper_bounds["id"] = "rw-tr-up-0"
    rwanda_tr_upper_bounds["split"] = "train"

    rwanda_bounds["id"] = "rw"
    rwanda_bounds["split"] = "all"

    uganda_te_bounds["id"] = "ug-te-0"
    uganda_te_bounds["split"] = "test"
    uganda_tr_bounds["id"] = "ug-tr-0"
    uganda_tr_bounds["split"] = "train"
    uganda_va_bounds["id"] = "ug-va-0"
    uganda_va_bounds["split"] = "val"
    uganda_bounds["id"] = "ug"
    uganda_bounds["split"] = "all"

    # concatenate all Rwanda bounds
    rw_all_bounds = pd.concat([
        rwanda_tr_lower_bounds, rwanda_va_bounds,
        rwanda_tr_upper_bounds, rwanda_bounds]).reset_index(drop=True)
    rw_all_bounds["Country"] = "Rwanda"

    # concatenate all Uganda bounds
    ug_all_bounds = pd.concat([
        uganda_te_bounds, uganda_tr_bounds, uganda_va_bounds,
        uganda_bounds]).reset_index(drop=True)
    ug_all_bounds["Country"] = "Uganda"

    # Concat all bounds and save to file
    all_bounds = pd.concat([rw_all_bounds, ug_all_bounds]).reset_index(
        drop=True)
    all_bounds.to_file(
        "./data/ground_truth/bounds_v2.geojson".format(tile_size))

    # get Uganda country bounds
    uganda_bounds = gpd.read_file("./data/country_masks/uganda.shp")
    # get data bounds for Rwanda and Uganda
    bounds = get_bounds()
    # intersect the polygon containing all bridge data with official Uganda
    # country bounds
    uganda_tr_va_te_bounds = get_polygon_gdf_from_bounds(bounds["uganda"])
    uganda_ssl_bounds = diff(uganda_bounds, uganda_tr_va_te_bounds)
    uganda_ssl_bounds["split"] = "ssl"
    uganda_ssl_bounds["id"] = "ug-ssl"
    uganda_ssl_bounds["Country"] = "Uganda"
    uganda_ssl_bounds["pos_neg"] = "all"

    # go through the v2 data
    for tile_size in [300, 600, 1200]:
        gt_fp = "./data/ground_truth/train_{}_v2.geojson".format(tile_size)
        # read the train data
        with open(gt_fp) as f:
            gdf = gpd.read_file(f)
        gdf = pd.concat([gdf, uganda_ssl_bounds]).reset_index(drop=True)
        gdf.to_file("./data/ground_truth/train_{}_v2_ssl.geojson".format(
            tile_size))

    # add uganda ssl to bounds
    uganda_ssl_bounds.drop(["pos_neg"], axis=1, inplace=True)
    with open("./data/ground_truth/bounds_v2.geojson") as f:
        gdf = gpd.read_file(f)
    gdf = pd.concat([gdf, uganda_ssl_bounds]).reset_index(drop=True)
    gdf.to_file("./data/ground_truth/bounds_v2_ssl.geojson".format(tile_size))
