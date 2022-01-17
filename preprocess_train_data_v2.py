import os.path as path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio

from rasterio.mask import mask
from shapely.geometry import Polygon

from src.data import get_square_area


def get_dfs():
    rwanda_df = pd.read_csv(
        "./data/ground_truth/"
        "Rwanda training data_AllSitesMinusVehicleBridges_21.11.05.csv")
    rwanda_df.dropna(inplace=True)
    print("Number of Rwanda data points: {}".format(len(rwanda_df)))

    uganda_df = pd.read_csv(
        "./data/ground_truth/Uganda_TrainingData_3districts_ADSK.csv")
    uganda_df.dropna(inplace=True)
    print("Number of Uganda data points: {}".format(len(uganda_df)))

    return rwanda_df, uganda_df


def get_bounds():

    _, uganda_df = get_dfs()

    lat_name = "GPS (Latitude)"
    lon_name = "GPS (Longitude)"

    bounds_dict = {c: {} for c in ["rwanda", "uganda"]}

    exclude_ids = ["1023076"]
    min_lat, max_lat = float("inf"), - float("inf")
    min_lon, max_lon = float("inf"), - float("inf")

    for i, row in uganda_df.iterrows():
        if row["Opportunity Unique Identifier"] in exclude_ids:
            continue
        lat = row[lat_name]
        lon = row[lon_name]
        min_lat = min(min_lat, lat)
        max_lat = max(max_lat, lat)
        min_lon = min(min_lon, lon)
        max_lon = max(max_lon, lon)

    min_lat -= 300 / 3600
    max_lat += 300 / 3600
    min_lon -= 300 / 3600
    max_lon += 300 / 3600

    bounds_dict["uganda"] = {
        "left": min_lon,
        "bottom": min_lat,
        "right": max_lon,
        "top": max_lat
    }

    rwanda_fp = ("./data/country_masks/rwanda_mask_1-3600.tiff")
    rwanda = rasterio.open(rwanda_fp)

    bounds_dict["rwanda"] = {
        "left": rwanda.bounds.left,
        "bottom": rwanda.bounds.bottom,
        "right": rwanda.bounds.right,
        "top": rwanda.bounds.top
    }
    return bounds_dict


def get_uganda_tr_range(bounds):
    tr_range = (
        bounds["uganda"]["left"] + (
            bounds["uganda"]["right"] - bounds["uganda"]["left"]) / 11 * 8,
        bounds["uganda"]["left"] + (
            bounds["uganda"]["right"] - bounds["uganda"]["left"]) / 11 * 10
    )
    return tr_range


def get_rwanda_va_range(bounds):
    val_range = (
        bounds["rwanda"]["left"] + (
            bounds["rwanda"]["right"] - bounds["rwanda"]["left"]) / 11 * 5,
        bounds["rwanda"]["left"] + (
            bounds["rwanda"]["right"] - bounds["rwanda"]["left"]) / 11 * 6
    )
    return val_range


if __name__ == "__main__":
    thres = 50

    lat_name = "GPS (Latitude)"
    lon_name = "GPS (Longitude)"
    crs = "EPSG:4326"
    bounds = get_bounds()
    va_range = get_rwanda_va_range(bounds)
    tr_range = get_uganda_tr_range(bounds)

    gpd_kwargs = {
        "crs": "EPSG:4326",
        "columns": ["geometry"]
    }

    rwanda_bounds = gpd.read_file("./data/country_masks/rwanda.shp")
    uganda_bounds = gpd.read_file("./data/country_masks/uganda.shp")

    # bounds for tr, va, te
    uganda_te_bounds = gpd.GeoDataFrame([
        Polygon([
            [bounds["uganda"]["left"], bounds["uganda"]["top"]],
            [tr_range[0], bounds["uganda"]["top"]],
            [tr_range[0], bounds["uganda"]["bottom"]],
            [bounds["uganda"]["left"], bounds["uganda"]["bottom"]]])],
        crs=crs, columns=["geometry"]
    )
    uganda_te_bounds = uganda_te_bounds.overlay(
        uganda_bounds, how="intersection")

    uganda_tr_bounds = gpd.GeoDataFrame([
        Polygon([
            [tr_range[0], bounds["uganda"]["top"]],
            [tr_range[1], bounds["uganda"]["top"]],
            [tr_range[1], bounds["uganda"]["bottom"]],
            [tr_range[0], bounds["uganda"]["bottom"]]
        ])], crs=crs, columns=["geometry"]
    )
    uganda_tr_bounds = uganda_tr_bounds.overlay(
        uganda_bounds, how="intersection")

    uganda_va_bounds = gpd.GeoDataFrame([
        Polygon([
            [tr_range[1], bounds["uganda"]["top"]],
            [bounds["uganda"]["right"], bounds["uganda"]["top"]],
            [bounds["uganda"]["right"], bounds["uganda"]["bottom"]],
            [tr_range[1], bounds["uganda"]["bottom"]]
        ])], **gpd_kwargs
    )
    uganda_va_bounds = uganda_va_bounds.overlay(
        uganda_bounds, how="intersection")

    rwanda_tr_lower_bounds = gpd.GeoDataFrame([
        Polygon([
            [bounds["rwanda"]["left"], bounds["rwanda"]["top"]],
            [va_range[0], bounds["rwanda"]["top"]],
            [va_range[0], bounds["rwanda"]["bottom"]],
            [bounds["rwanda"]["left"], bounds["rwanda"]["bottom"]]])],
        crs=crs, columns=["geometry"]
    )
    rwanda_tr_lower_bounds = rwanda_tr_lower_bounds.overlay(
        rwanda_bounds, how="intersection")

    rwanda_va_bounds = gpd.GeoDataFrame([
        Polygon([
            [va_range[0], bounds["rwanda"]["top"]],
            [va_range[1], bounds["rwanda"]["top"]],
            [va_range[1], bounds["rwanda"]["bottom"]],
            [va_range[0], bounds["rwanda"]["bottom"]]
        ])], crs=crs, columns=["geometry"]
    )
    rwanda_va_bounds = rwanda_va_bounds.overlay(
        rwanda_bounds, how="intersection")

    rwanda_tr_upper_bounds = gpd.GeoDataFrame([
        Polygon([
            [va_range[1], bounds["rwanda"]["top"]],
            [bounds["rwanda"]["right"], bounds["rwanda"]["top"]],
            [bounds["rwanda"]["right"], bounds["rwanda"]["bottom"]],
            [va_range[1], bounds["rwanda"]["bottom"]]
        ])], **gpd_kwargs
    )
    rwanda_tr_upper_bounds = rwanda_tr_upper_bounds.overlay(
        rwanda_bounds, how="intersection")

    for tile_size in [300, 600, 1200]:
        rwanda_df, uganda_df = get_dfs()

        pos_size = tile_size - thres
        non_neg_size = 2 * tile_size - 50
        # rwanda
        recid_name = "Record ID"

        rwanda_df[recid_name] = "rw-" + rwanda_df[recid_name]
        rwanda_df = rwanda_df.rename(columns={recid_name: "id"})
        rwanda_df.drop(["Site Name"], inplace=True, axis=1)

        # positive area
        rwanda_df["pos_polygon"] = rwanda_df.apply(
            lambda x: Polygon(
                get_square_area(x[lon_name], x[lat_name],
                                square_length=pos_size)),
            axis=1)
        # non negative area
        rwanda_df["non_neg_polygon"] = rwanda_df.apply(
            lambda x: Polygon(
                get_square_area(
                    x[lon_name], x[lat_name], square_length=non_neg_size)),
            axis=1)

        rwanda_df.loc[
            rwanda_df[lon_name] < va_range[0], "split"] = "train_lower"
        rwanda_df.loc[rwanda_df[lon_name] < va_range[0], "min_x"] = bounds[
            "rwanda"]["left"]
        rwanda_df.loc[rwanda_df[lon_name] < va_range[0], "max_x"] = va_range[0]

        rwanda_df.loc[(rwanda_df[lon_name] >= va_range[0]) &
                      (rwanda_df[lon_name] < va_range[1]), "split"] = "val"
        rwanda_df.loc[
            (rwanda_df[lon_name] >= va_range[0]) &
            (rwanda_df[lon_name] < va_range[1]), "min_x"] = va_range[0]
        rwanda_df.loc[
            (rwanda_df[lon_name] >= va_range[0]) &
            (rwanda_df[lon_name] < va_range[1]), "max_x"] = va_range[1]
        rwanda_df.loc[rwanda_df[lon_name] >=
                      va_range[1], "split"] = "train_upper"
        rwanda_df.loc[rwanda_df[lon_name] >=
                      va_range[1], "min_x"] = va_range[1]
        rwanda_df.loc[rwanda_df[lon_name] >= va_range[1], "max_x"] = bounds[
            "rwanda"]["right"]

        recid_name = "Opportunity Unique Identifier"
        uganda_df[recid_name] = "ug-" + uganda_df[recid_name]
        uganda_df = uganda_df.rename(columns={recid_name: "id"})
        uganda_df.drop(["Opportunity ID"], inplace=True, axis=1)

        uganda_df["pos_polygon"] = uganda_df.apply(
            lambda x: Polygon(get_square_area(
                x[lon_name], x[lat_name], square_length=pos_size)), axis=1)
        uganda_df["non_neg_polygon"] = uganda_df.apply(
            lambda x: Polygon(
                get_square_area(
                    x[lon_name], x[lat_name], square_length=non_neg_size)),
            axis=1)

        uganda_df.loc[uganda_df[lon_name] < tr_range[0], "split"] = "test"
        uganda_df.loc[uganda_df[lon_name] < va_range[0], "min_x"] = bounds[
            "uganda"]["left"]
        uganda_df.loc[uganda_df[lon_name] < va_range[0], "max_x"] = tr_range[0]

        uganda_df.loc[(uganda_df[lon_name] >= tr_range[0]) &
                      (uganda_df[lon_name] < tr_range[1]), "split"] = "train"
        uganda_df.loc[
            (uganda_df[lon_name] >= tr_range[0]) &
            (uganda_df[lon_name] < tr_range[1]), "min_x"] = tr_range[0]
        uganda_df.loc[
            (uganda_df[lon_name] >= tr_range[0]) &
            (uganda_df[lon_name] < tr_range[1]), "max_x"] = tr_range[1]
        uganda_df.loc[uganda_df[lon_name] >= tr_range[1], "split"] = "val"
        uganda_df.loc[
            uganda_df[lon_name] >= tr_range[1], "min_x"] = tr_range[1]
        uganda_df.loc[uganda_df[lon_name] >= tr_range[1], "max_x"] = bounds[
            "uganda"]["right"]

        rwanda_tr_lower_df = gpd.GeoDataFrame(
            rwanda_df[rwanda_df.split == "train_lower"].drop(
                "pos_polygon", axis=1).rename(
                    {"non_neg_polygon": "geometry"}, axis=1),
            crs="EPSG:4326"
        )
        rwanda_va_df = gpd.GeoDataFrame(
            rwanda_df[rwanda_df.split == "val"].drop(
                "pos_polygon", axis=1).rename(
                    {"non_neg_polygon": "geometry"}, axis=1),
            crs="EPSG:4326"
        )
        rwanda_tr_upper_df = gpd.GeoDataFrame(
            rwanda_df[rwanda_df.split == "train_upper"].drop(
                "pos_polygon", axis=1).rename(
                    {"non_neg_polygon": "geometry"}, axis=1),
            crs="EPSG:4326"
        )

        uganda_te_df = gpd.GeoDataFrame(
            uganda_df[uganda_df.split == "test"].drop(
                "pos_polygon", axis=1).rename(
                    {"non_neg_polygon": "geometry"}, axis=1),
            crs="EPSG:4326"
        )
        uganda_tr_df = gpd.GeoDataFrame(
            uganda_df[uganda_df.split == "train"].drop(
                "pos_polygon", axis=1).rename(
                    {"non_neg_polygon": "geometry"}, axis=1),
            crs="EPSG:4326"
        )
        uganda_va_df = gpd.GeoDataFrame(
            uganda_df[uganda_df.split == "val"].drop(
                "pos_polygon", axis=1).rename(
                    {"non_neg_polygon": "geometry"}, axis=1),
            crs="EPSG:4326"
        )

        rwanda_df["Country"] = "Rwanda"
        uganda_df["Country"] = "Uganda"

        poss = pd.concat([rwanda_df, uganda_df]).drop(
            "non_neg_polygon", axis=1).rename(
                {"pos_polygon": "geometry"}, axis=1)
        poss["pos_neg"] = "pos"

        neg_rwanda_tr_lower = rwanda_tr_lower_bounds.overlay(
            rwanda_tr_lower_df, how="difference")
        neg_rwanda_tr_upper = rwanda_tr_upper_bounds.overlay(
            rwanda_tr_upper_df, how="difference")
        neg_rwanda_val = rwanda_va_bounds.overlay(
            rwanda_va_df, how="difference")

        neg_rwanda_tr_lower = gpd.GeoDataFrame(
            neg_rwanda_tr_lower, **gpd_kwargs)
        neg_rwanda_tr_upper = gpd.GeoDataFrame(
            neg_rwanda_tr_upper, **gpd_kwargs)
        neg_rwanda_val = gpd.GeoDataFrame(
            neg_rwanda_val, **gpd_kwargs)

        neg_rwanda_tr_lower = neg_rwanda_tr_lower.reset_index().rename(
            {"index": "id"}, axis=1)
        neg_rwanda_tr_lower["id"] = neg_rwanda_tr_lower["id"].astype(str)
        neg_rwanda_tr_lower["id"] = "rw-neg-tr-lower" + \
            neg_rwanda_tr_lower["id"]
        neg_rwanda_tr_lower["min_x"] = bounds["rwanda"]["left"]
        neg_rwanda_tr_lower["max_x"] = va_range[0]
        neg_rwanda_tr_lower["split"] = "train_lower"
        neg_rwanda_tr_lower["Country"] = "Rwanda"

        neg_rwanda_tr_upper = neg_rwanda_tr_upper.reset_index().rename(
            {"index": "id"}, axis=1)
        neg_rwanda_tr_upper["id"] = neg_rwanda_tr_upper["id"].astype(str)
        neg_rwanda_tr_upper["id"] = "rw-neg-tr-upper" + \
            neg_rwanda_tr_upper["id"]
        neg_rwanda_tr_upper["min_x"] = va_range[1]
        neg_rwanda_tr_upper["max_x"] = bounds["rwanda"]["right"]
        neg_rwanda_tr_upper["split"] = "train_upper"
        neg_rwanda_tr_upper["Country"] = "Rwanda"

        neg_rwanda_val = neg_rwanda_val.reset_index().rename(
            {"index": "id"}, axis=1)
        neg_rwanda_val["id"] = neg_rwanda_val["id"].astype(str)
        neg_rwanda_val["id"] = "rw-neg-val-" + neg_rwanda_val["id"]
        neg_rwanda_val["min_x"] = va_range[0]
        neg_rwanda_val["max_x"] = va_range[1]
        neg_rwanda_val["split"] = "val"
        neg_rwanda_val["Country"] = "Rwanda"

        neg_uganda_te = uganda_te_bounds.overlay(
            uganda_te_df, how="difference")
        neg_uganda_tr = uganda_tr_bounds.overlay(
            uganda_tr_df, how="difference")
        neg_uganda_va = uganda_va_bounds.overlay(
            uganda_va_df, how="difference")

        neg_uganda_te = gpd.GeoDataFrame(neg_uganda_te, **gpd_kwargs)
        neg_uganda_tr = gpd.GeoDataFrame(neg_uganda_tr, **gpd_kwargs)
        neg_uganda_va = gpd.GeoDataFrame(neg_uganda_va, **gpd_kwargs)

        neg_uganda_te = neg_uganda_te.reset_index().rename(
            {"index": "id"}, axis=1)
        neg_uganda_te["id"] = neg_uganda_te["id"].astype(str)
        neg_uganda_te["id"] = "ug-neg-te-lower" + neg_uganda_te["id"]
        neg_uganda_te["min_x"] = bounds["uganda"]["left"]
        neg_uganda_te["max_x"] = tr_range[0]
        neg_uganda_te["split"] = "test"
        neg_uganda_te["Country"] = "Uganda"

        neg_uganda_tr = neg_uganda_tr.reset_index().rename(
            {"index": "id"}, axis=1)
        neg_uganda_tr["id"] = neg_uganda_tr["id"].astype(str)
        neg_uganda_tr["id"] = "ug-neg-tr-" + neg_uganda_tr["id"]
        neg_uganda_tr["min_x"] = tr_range[0]
        neg_uganda_tr["max_x"] = tr_range[1]
        neg_uganda_tr["split"] = "train"
        neg_uganda_tr["Country"] = "Uganda"

        neg_uganda_va = neg_uganda_va.reset_index().rename(
            {"index": "id"}, axis=1)
        neg_uganda_va["id"] = neg_uganda_va["id"].astype(str)
        neg_uganda_va["id"] = "ug-neg-te-upper" + neg_uganda_va["id"]
        neg_uganda_va["min_x"] = tr_range[1]
        neg_uganda_va["max_x"] = bounds["uganda"]["right"]
        neg_uganda_va["split"] = "val"
        neg_uganda_va["Country"] = "Uganda"

        negs = pd.concat([neg_rwanda_tr_lower, neg_rwanda_tr_upper,
                          neg_rwanda_val, neg_uganda_te,
                          neg_uganda_va, neg_uganda_tr])
        negs["pos_neg"] = "neg"
        df_pos_neg = gpd.GeoDataFrame(
            pd.concat([poss, negs]).reset_index(drop=True), crs=crs)
        df_pos_neg.to_file(
            "./data/ground_truth/train_{}_v2.geojson".format(tile_size))
