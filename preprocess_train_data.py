import geopandas as gpd
import pandas as pd
import rasterio

from shapely.geometry import Polygon

from src.data_epsg_4326 import get_square_area


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

    rwanda_fp = ("./data/country_masks/used_for_pre_processing/"
                 "rwanda_mask_1-3600.tiff")
    rwanda = rasterio.open(rwanda_fp)

    bounds_dict["rwanda"] = {
        "left": rwanda.bounds.left,
        "bottom": rwanda.bounds.bottom,
        "right": rwanda.bounds.right,
        "top": rwanda.bounds.top
    }
    return bounds_dict


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

    rwanda_bounds = gpd.read_file(
        "./data/country_masks/used_for_pre_processing/rwanda.shp")
    uganda_bounds = gpd.read_file(
        "./data/country_masks/used_for_pre_processing/uganda.shp")

    # bounds for tr, va, te
    uganda_te_bounds = gpd.GeoDataFrame([
        Polygon([
            [bounds["uganda"]["left"], bounds["uganda"]["top"]],
            [bounds["uganda"]["right"], bounds["uganda"]["top"]],
            [bounds["uganda"]["right"], bounds["uganda"]["bottom"]],
            [bounds["uganda"]["left"], bounds["uganda"]["bottom"]]])],
        crs=crs, columns=["geometry"]
    )
    uganda_te_bounds = uganda_bounds.overlay(
        uganda_te_bounds, how="intersection")

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
        ])], crs="EPSG:4326", columns=["geometry"]
    )
    rwanda_tr_upper_bounds = rwanda_tr_upper_bounds.overlay(
        rwanda_bounds, how="intersection")

    # one polygon to show space of all positive points for sampling
    # = tile_size - 50
    # one polygon to show space of where negative points for sampling cannot be
    # = tile_size - 50 + tile_size / 0.5 + tile_size / 0.5
    # = tile_size - 50 + tile_size
    # = 2 * tile_size - 50

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

        rwanda_df.loc[rwanda_df[lon_name] <
                      va_range[0], "split"] = "train_lower"
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
                x[lon_name], x[lat_name], square_length=pos_size)),
            axis=1)
        uganda_df["non_neg_polygon"] = uganda_df.apply(
            lambda x: Polygon(
                get_square_area(
                    x[lon_name], x[lat_name], square_length=non_neg_size)),
            axis=1)
        uganda_df["split"] = "test"
        uganda_df["min_x"] = bounds["uganda"]["left"]
        uganda_df["max_x"] = bounds["uganda"]["right"]

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

        neg_uganda_te = gpd.GeoDataFrame(
            uganda_df.drop("pos_polygon", axis=1).rename(
                {"non_neg_polygon": "geometry"}, axis=1),
            crs="EPSG:4326"
        )
        rwanda_df["Country"] = "Rwanda"
        uganda_df["Country"] = "Uganda"
        df = pd.concat([rwanda_df, uganda_df]).drop(
            "non_neg_polygon", axis=1).rename(
                {"pos_polygon": "geometry"}, axis=1)
        df["pos_neg"] = "pos"

        neg_rwanda_tr_lower = rwanda_tr_lower_bounds.overlay(
            rwanda_tr_lower_df, how="difference")
        neg_rwanda_tr_upper = rwanda_tr_upper_bounds.overlay(
            rwanda_tr_upper_df, how="difference")

        neg_rwanda_tr_lower = gpd.GeoDataFrame(
            neg_rwanda_tr_lower, crs="EPSG:4326", columns=["geometry"])
        neg_rwanda_tr_upper = gpd.GeoDataFrame(
            neg_rwanda_tr_upper, crs="EPSG:4326", columns=["geometry"])

        neg_rwanda_tr_lower = neg_rwanda_tr_lower.reset_index().rename(
            {"index": "id"}, axis=1)
        neg_rwanda_tr_lower["id"] = neg_rwanda_tr_lower["id"].astype(str)
        neg_rwanda_tr_lower["id"] = "rw-neg-tr-lower" + \
            neg_rwanda_tr_lower["id"]
        neg_rwanda_tr_lower["min_x"] = bounds["rwanda"]["left"]
        neg_rwanda_tr_lower["max_x"] = va_range[0]
        neg_rwanda_tr_lower["split"] = "train-lower"
        neg_rwanda_tr_lower["Country"] = "Rwanda"

        neg_rwanda_tr_upper = neg_rwanda_tr_upper.reset_index().rename(
            {"index": "id"}, axis=1)
        neg_rwanda_tr_upper["id"] = neg_rwanda_tr_upper["id"].astype(str)
        neg_rwanda_tr_upper["id"] = "rw-neg-tr-upper" + \
            neg_rwanda_tr_upper["id"]
        neg_rwanda_tr_upper["min_x"] = va_range[1]
        neg_rwanda_tr_upper["max_x"] = bounds["rwanda"]["right"]
        neg_rwanda_tr_upper["split"] = "train-upper"
        neg_rwanda_tr_upper["Country"] = "Rwanda"

        neg_rwanda_val = rwanda_va_bounds.overlay(
            rwanda_va_df, how="difference")
        neg_rwanda_val = neg_rwanda_val.drop(
            "id", axis=1).reset_index().rename({"index": "id"}, axis=1)
        neg_rwanda_val["id"] = neg_rwanda_val["id"].astype(str)
        neg_rwanda_val["id"] = "rw-neg-val-" + neg_rwanda_val["id"]
        neg_rwanda_val["min_x"] = va_range[0]
        neg_rwanda_val["max_x"] = va_range[1]
        neg_rwanda_val["split"] = "val"
        neg_rwanda_val["Country"] = "Rwanda"

        neg_uganda_te = uganda_bounds.overlay(neg_uganda_te, how="difference")
        neg_uganda_te = neg_uganda_te.drop(
            "id", axis=1).reset_index().rename({"index": "id"}, axis=1)
        neg_uganda_te["id"] = neg_uganda_te["id"].astype(str)
        neg_uganda_te["id"] = "ug-neg-te-" + neg_uganda_te["id"]
        neg_uganda_te["min_x"] = bounds["uganda"]["left"]
        neg_uganda_te["max_x"] = bounds["uganda"]["right"]
        neg_uganda_te["split"] = "test"
        neg_uganda_te["Country"] = "Uganda"
        negs = pd.concat([neg_rwanda_tr_lower, neg_rwanda_tr_upper,
                          neg_rwanda_val, neg_uganda_te])
        negs["pos_neg"] = "neg"
        df_pos_neg = gpd.GeoDataFrame(
            pd.concat([df, negs]).reset_index(drop=True), crs=crs)
        df_pos_neg.to_file(
            "./data/ground_truth/v2/train_{}.geojson".format(tile_size))
