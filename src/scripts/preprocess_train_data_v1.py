"""
Data V1 splits Rwanda into 11 parts equally by its longitude. Parts 1-5 and
7-11 are used for training, part 6 is used for validation, and Uganda is only
used for testing.

For the semi-supervised learning (ssl) version, we use the Uganda data without
any labels.
"""
from typing import List
import geopandas as gpd
import pandas as pd
import rasterio

from shapely.geometry import Polygon

from src.data import geometry

THRES = 50  # meters
CRS = "EPSG:4326"
GDF_KWARGS = {"crs": CRS, "columns": ["geometry"]}


def get_dfs():
    """Reads the CSV data from Bridges To Prosperity containing bridge sites.

    rwanda_df has the following columns:
        Record ID,Site Name,GPS (Latitude),GPS (Longitude).

    uganda_df has the following columns:
        Opportunity Unique Identifier,Opportunity ID,GPS (Latitude),
        GPS (Longitude).
    Returns:
        df (Tuple[DataFrame]): Tuple with Rwanda and Uganda CSV data in a
            Pandas DataFrame each.

    """
    rwanda_df = pd.read_csv(
        "./data/ground_truth/"
        "Rwanda training data_AllSitesMinusVehicleBridges_21.11.05.csv")
    rwanda_df.dropna(inplace=True)
    print("Number of Rwanda data points: {}".format(len(rwanda_df)))

    uganda_df = pd.read_csv(
        "./data/ground_truth/Uganda_TrainingData_3districts_ADSK.csv")
    uganda_df.dropna(inplace=True)
    print("Number of Uganda data points: {}".format(len(uganda_df)))
    df = (rwanda_df, uganda_df)
    return df


def get_bounds(uganda_df: pd.DataFrame = get_dfs()[1],
               exclude_ids: List = ["1023076"],
               lat_name: str = "GPS (Latitude)",
               lon_name: str = "GPS (Longitude)"):
    """Calculate country bounds for Rwanda and Uganda.

    For Uganda, we only use the part of Uganda that has bridge sites. For
    Rwanda the country bounds are used.

    Returns:
        bounds_dict (Dict): Dictionary with country as keys (`uganda`,
            `rwanda`) and values are dictionaries containing the boundaries (
            each dictionary has four keys `left`, `bottom`, `right`, `top`).
    """

    bounds_dict = {c: {} for c in ["rwanda", "uganda"]}

    # wrong gps coordinate in this row

    min_lat, max_lat = float("inf"), - float("inf")
    min_lon, max_lon = float("inf"), - float("inf")

    # go through data and estimate min and max latitude and longitude
    for _, row in uganda_df.iterrows():
        if row["Opportunity Unique Identifier"] in exclude_ids:
            continue
        lat = row[lat_name]
        lon = row[lon_name]
        min_lat = min(min_lat, lat)
        max_lat = max(max_lat, lat)
        min_lon = min(min_lon, lon)
        max_lon = max(max_lon, lon)

    # add / substract to min and max latitude and longitude each 3-arc seconds
    min_lat -= 300 / 3600
    max_lat += 300 / 3600
    min_lon -= 300 / 3600
    max_lon += 300 / 3600

    # use these bounds for uganda
    bounds_dict["uganda"] = {
        "left": min_lon,
        "bottom": min_lat,
        "right": max_lon,
        "top": max_lat
    }

    rwanda_fp = ("./data/country_masks/rwanda_mask_1-3600.tiff")
    rwanda = rasterio.open(rwanda_fp)

    # use the country bounds for rwanda
    bounds_dict["rwanda"] = {
        "left": rwanda.bounds.left,
        "bottom": rwanda.bounds.bottom,
        "right": rwanda.bounds.right,
        "top": rwanda.bounds.top
    }
    return bounds_dict


def get_rwanda_va_range(bounds):
    """Getting the longitude range for Rwanda validation set.
    We are splitting Rwanda into 11 parts. The sixth part is used for
    validation.

    Args:
        bounds (Dict): Contains the bounds for Rwanda.

    Returns:
        val_range (Tuple): The tuple consists of two latitude values
            representing the minimum and maximum latitude values that the
            validation data can have.
    """
    # we split the country Rwanda vertically according to
    val_range = (
        bounds["rwanda"]["left"] + (
            bounds["rwanda"]["right"] - bounds["rwanda"]["left"]) / 11 * 5,
        bounds["rwanda"]["left"] + (
            bounds["rwanda"]["right"] - bounds["rwanda"]["left"]) / 11 * 6
    )
    return val_range


def get_polygon_gdf_from_bounds(polygon_bounds, gdf_kwargs=GDF_KWARGS):
    """Returns a GeoDataFrame with polygon given bounds of a region.

    Args:
        polygon_bounds (Dict): Contains keys `left`, `bottom`, `right`, `top`.
        gdf_kwargs (Dict, optional): Dictionary containing settings for
            Coordinate Reference System (crs) and geometry.

    Returns:
        polygon_gdf (geopandas.GeoDataFrame): A GeoDataFrame with one entry
            which is the polygon created from the bounds.
    """
    polygon = Polygon([
        [polygon_bounds["left"], polygon_bounds["top"]],
        [polygon_bounds["right"], polygon_bounds["top"]],
        [polygon_bounds["right"], polygon_bounds["bottom"]],
        [polygon_bounds["left"], polygon_bounds["bottom"]]])
    polygon_gdf = gpd.GeoDataFrame([polygon], **GDF_KWARGS)
    return polygon_gdf


def set_split_and_min_max_vals(df: gpd.GeoDataFrame, split_name: str,
                               lon_name: str, left_bound: float,
                               right_bound: float):
    """Set a name, e.g., `train`, to column `split` and set columns `min_x`
        and `max_x` to left and right bounds.

    Args:
        df (gpd.GeoDataFrame): GeoPandasDataFrame to be edited.
        split_name (str): The name for this split of data.
        lon_name (str): Name of column containing the longitude.
        left_bound (float): Left bound.
        right_bound (float): Right bound.
    """
    constraint = (df[lon_name] >= left_bound) & (df[lon_name] < right_bound)
    df.loc[constraint, "split"] = split_name
    df.loc[constraint, "min_x"] = left_bound
    df.loc[constraint, "max_x"] = right_bound
    return df


def diff(df1: gpd.GeoDataFrame, df2: gpd.GeoDataFrame):
    """To obtain the geometries that are part of df1 but are not in df2.

    Args:
        df1 (gpd.GeoDataFrame) 
        df2 (gpd.GeoDataFrame)
    Returns: Difference of df1 and df2.
    """
    return df1.overlay(df2, how="difference").drop(["id"], axis=1)


def intersect(df1: gpd.GeoDataFrame, df2: gpd.GeoDataFrame):
    """To obtain the geometries that are part of df1 and df2.

    Args:
        df1 (gpd.GeoDataFrame) 
        df2 (gpd.GeoDataFrame)
    Returns: Intersection of df1 and df2.
    """
    return df1.overlay(df2, how="intersection")


def format_neg_entries(df: gpd.GeoDataFrame, id_prefix: str, min_x: float,
                       max_x: float, split_name: str, country: str):
    """Formats negative entry.
    """
    df = df.reset_index().rename({"index": "id"}, axis=1)
    df["id"] = df["id"].astype(str)
    df["id"] = id_prefix + df["id"]
    df["min_x"] = min_x
    df["max_x"] = max_x
    df["split"] = split_name
    df["Country"] = country
    return df


if __name__ == "__main__":
    lat_name = "GPS (Latitude)"
    lon_name = "GPS (Longitude)"

    # get data bounds for Rwanda and Uganda
    bounds = get_bounds()
    # get validation range for Rwanda
    va_range = get_rwanda_va_range(bounds)

    # loading shape files containing country bounds
    rwanda_bounds = gpd.read_file("./data/country_masks/rwanda.shp")
    uganda_bounds = gpd.read_file("./data/country_masks/uganda.shp")

    # intersect the polygon containing all bridge data with official Uganda
    # country bounds
    uganda_te_bounds = get_polygon_gdf_from_bounds(bounds["uganda"])
    uganda_te_bounds = intersect(uganda_bounds, uganda_te_bounds)

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
        # get bridge site from B2P
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

        uganda_df["split"] = "test"
        uganda_df["min_x"] = bounds["uganda"]["left"]
        uganda_df["max_x"] = bounds["uganda"]["right"]
        # for each set (train_lower, val, train_upper, test), drop pos_polygon
        # and rename non_negative_polygon to geometry
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
        nn_uganda_te = gpd.GeoDataFrame(
            uganda_df.drop("pos_polygon", axis=1).rename(
                {"non_neg_polygon": "geometry"}, axis=1), crs=CRS)

        # the area in which center points for `negative` tiles is obtained by
        # by calculating the geometries that are part of the bounds but are not
        # contained in non negative geometry (== difference)
        neg_rwanda_tr_lower = diff(rwanda_tr_lower_bounds, nn_rwanda_tr_lower)
        neg_rwanda_va = diff(rwanda_va_bounds, nn_rwanda_va)
        neg_rwanda_tr_upper = diff(rwanda_tr_upper_bounds, nn_rwanda_tr_upper)
        neg_uganda_te = diff(uganda_te_bounds, nn_uganda_te)

        # set countries
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
            bounds["uganda"]["right"], "test", "Uganda")

        # concatenate all negative entries
        negs = pd.concat([neg_rwanda_tr_lower, neg_rwanda_tr_upper,
                          neg_rwanda_va, neg_uganda_te])
        negs["pos_neg"] = "neg"
        # concatenate positive and negative entries
        df_pos_neg = gpd.GeoDataFrame(
            pd.concat([df, negs]).reset_index(drop=True), crs=CRS)
        # save to file
        df_pos_neg.to_file(
            "./data/ground_truth/train_{}_v1.geojson".format(tile_size))

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
    uganda_bounds["id"] = "ug"
    uganda_bounds["split"] = "all"

    # concatenate all Rwanda bounds
    rw_all_bounds = pd.concat([
        rwanda_tr_lower_bounds, rwanda_va_bounds,
        rwanda_tr_upper_bounds, rwanda_bounds]).reset_index(drop=True)
    rw_all_bounds["Country"] = "Rwanda"

    # concatenate all Uganda bounds
    ug_all_bounds = pd.concat([
        uganda_te_bounds, uganda_bounds]).reset_index(drop=True)
    ug_all_bounds["Country"] = "Uganda"

    # Concat all bounds and save to file
    all_bounds = pd.concat([rw_all_bounds, ug_all_bounds]).reset_index(
        drop=True)
    all_bounds.to_file(
        "./data/ground_truth/bounds_v1.geojson".format(tile_size))

    # load uganda bounds
    uganda_bounds = gpd.read_file("./data/country_masks/uganda.shp")
    uganda_bounds["split"] = "ssl"
    uganda_bounds["id"] = "ug-ssl"
    uganda_bounds["Country"] = "Uganda"
    uganda_bounds["pos_neg"] = "all"

    # go through the v1 data
    for tile_size in [300, 600, 1200]:
        gt_fp = "./data/ground_truth/train_{}_v1.geojson".format(tile_size)
        # read the train data
        with open(gt_fp) as f:
            gdf = gpd.read_file(f)
        gdf = pd.concat([gdf, uganda_bounds]).reset_index(drop=True)
        gdf.to_file("./data/ground_truth/train_{}_v1_ssl.geojson".format(
            tile_size))

    # add uganda ssl to bounds
    uganda_bounds = gpd.read_file("./data/country_masks/uganda.shp")
    uganda_bounds["split"] = "ssl"
    uganda_bounds["id"] = "ug-ssl"
    uganda_bounds["Country"] = "Uganda"
    with open("./data/ground_truth/bounds_v1.geojson") as f:
        gdf = gpd.read_file(f)
    gdf = pd.concat([gdf, uganda_bounds]).reset_index(drop=True)
    gdf.to_file("./data/ground_truth/bounds_v1_ssl.geojson".format(
        tile_size))
