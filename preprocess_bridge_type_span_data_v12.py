from typing import Tuple
import uuid
import pandas as pd
import geopandas as gpd

from shapely.geometry import Point

import rasterio

from preprocess_train_data_v1 import get_rwanda_va_range
from preprocess_train_data_v2 import get_uganda_tr_range


BRIDGE_TYPE_SPAN_DATA = (
    "data/bridge_type_span_data/data_bridge_type_span_estimation_clean.csv")


def preprocess_data(data_fp: str = BRIDGE_TYPE_SPAN_DATA) -> pd.DataFrame:
    """Reads bridge type and span data and preprocess it."""
    rwanda_bounds = gpd.read_file("./data/country_masks/rwanda.shp")
    uganda_bounds = gpd.read_file("./data/country_masks/uganda.shp")

    df = pd.read_csv(data_fp)

    df["Opportunity Unique Identifier"] = df.apply(
        lambda x: uuid.uuid5(
            uuid.NAMESPACE_URL, x["Bridge Opportunity: Opportunity Name"]),
        axis=1)
    df["Country"] = df.apply(
        lambda x: x["Bridge Opportunity: Opportunity Name"].split(" ")[0],
        axis=1)

    df = df[(df.Country == "Uganda") | (df.Country == "Rwanda")]
    df.reset_index(inplace=True, drop=True)

    df["in_Country"] = df.apply(
        lambda x: (
            uganda_bounds.iloc[0].geometry.contains(Point(
                x["Bridge Opportunity: GPS (Longitude)"],
                x["Bridge Opportunity: GPS (Latitude)"]
            ))
            if x.Country == "Uganda"
            else rwanda_bounds.iloc[0].geometry.contains(Point(
                x["Bridge Opportunity: GPS (Longitude)"],
                x["Bridge Opportunity: GPS (Latitude)"]
            ))
        ),
        axis=1
    )

    df.loc[
        (df["Bridge Opportunity: Opportunity Name"] ==
         "Rwanda - Karushuga - 1014260")
        & (df["in_Country"] == False), "in_Country"] = True

    df.loc[
        (df["Bridge Opportunity: Opportunity Name"] ==
         "Rwanda - Kinamba - 1013571")
        & (df["in_Country"] == False), "in_Country"] = True

    df.loc[
        (df["Bridge Opportunity: Opportunity Name"] ==
         "Rwanda - Runyeshyanga - 1013572")
        & (df["in_Country"] == False), "in_Country"] = True

    df.loc[
        (df["Bridge Opportunity: Opportunity Name"] ==
         "Rwanda - Rwamuzenga - 1013575")
        & (df["in_Country"] == False), "in_Country"] = True

    df.loc[
        (df["Bridge Opportunity: Opportunity Name"] ==
         "Rwanda - Rwihinda - 1013573")
        & (df["in_Country"] == False), "in_Country"] = True

    df.loc[
        (df["Bridge Opportunity: Opportunity Name"] ==
         "Rwanda - Akagera - 1014407")
        & (df["in_Country"] == False), "in_Country"] = True

    df.loc[
        (df["Bridge Opportunity: Opportunity Name"] ==
         "Rwanda - Fotorero - 1014404")
        & (df["in_Country"] == False), "in_Country"] = True

    df.loc[
        (df["Bridge Opportunity: Opportunity Name"] ==
         "Uganda - Kaptolomongon Bridge - 1011061")
        & (df["in_Country"] == False), "in_Country"] = True

    df = df.drop(df[df.in_Country == False].index)

    return df


def get_bounds(df: pd.DataFrame,
               lat_name: str = "Bridge Opportunity: GPS (Latitude)",
               lon_name: str = "Bridge Opportunity: GPS (Longitude)"
               ) -> Tuple[float]:
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
    for _, row in df[df.Country == "Uganda"].iterrows():
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


if __name__ == "__main__":
    # v1
    lat_name = "Bridge Opportunity: GPS (Latitude)",
    lon_name = "Bridge Opportunity: GPS (Longitude)"
    df = preprocess_data()
    bounds = get_bounds(df)
    va_range = get_rwanda_va_range(bounds)
    df.loc[
        (df.Country == "Rwanda") & (df[lon_name] < va_range[0]),
        "split"] = "train-lower"
    df.loc[
        (df.Country == "Rwanda") & (va_range[0] <= df[lon_name])
        & (df[lon_name] < va_range[1]),
        "split"] = "val"

    df.loc[
        (df.Country == "Rwanda") & (va_range[1] <= df[lon_name]),
        "split"] = "train-upper"
    df.loc[df.Country == "Uganda", "split"] = "test"

    df.to_csv("data/bridge_type_span_data/data_v1.csv", index=False)
    print("Data version 1:")
    print("Train: {} Val: {} Test:{}".format(
        len(df[df.split.str.startswith("train")]),
        len(df[df.split.str.startswith("val")]),
        len(df[df.split.str.startswith("test")])))

    # v2
    df = preprocess_data()
    bounds = get_bounds(df)
    rw_va_range = get_rwanda_va_range(bounds)
    ug_tr_range = get_uganda_tr_range(bounds)

    df.loc[
        (df.Country == "Rwanda") & (df[lon_name] < rw_va_range[0]),
        "split"] = "train-lower"
    df.loc[
        (df.Country == "Rwanda") & (rw_va_range[0] <= df[lon_name])
        & (df[lon_name] < rw_va_range[1]),
        "split"] = "val-rw"
    df.loc[
        (df.Country == "Rwanda") & (rw_va_range[1] <= df[lon_name]),
        "split"] = "train-upper"

    df.loc[
        (df.Country == "Uganda") & (df[lon_name] < ug_tr_range[0]),
        "split"] = "test"
    df.loc[
        (df.Country == "Uganda") & (ug_tr_range[0] <= df[lon_name])
        & (df[lon_name] < ug_tr_range[1]),
        "split"] = "train"
    df.loc[
        (df.Country == "Uganda") & (ug_tr_range[1] <= df[lon_name]),
        "split"] = "val-ug"

    df.to_csv("data/bridge_type_span_data/data_v2.csv", index=False)
    print("Data version 2:")
    print("Train: {} Val: {} Test:{}".format(
        len(df[df.split.str.startswith("train")]),
        len(df[df.split.str.startswith("val")]),
        len(df[df.split.str.startswith("test")])))
