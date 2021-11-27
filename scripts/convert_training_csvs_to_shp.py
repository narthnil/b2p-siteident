import geopandas as gpd
import pandas as pd
from shapely.geometry import Point


rwanda_training_file = (
    "data/Rwanda training data_AllSitesMinusVehicleBridges_21.11.05.csv"
)
uganda_training_file = "data/Uganda_TrainingData_3districts_ADSK.csv"


def main():
    rwanda_data = pd.read_csv(rwanda_training_file)
    print(rwanda_data.columns)
    rwanda_data["geometry"] = rwanda_data.apply(
        lambda row: Point(row["GPS (Longitude)"], row["GPS (Latitude)"]), axis=1
    )
    rwanda_geo_data = gpd.GeoDataFrame(rwanda_data, crs="EPSG:4326")
    rwanda_geo_data.to_file(rwanda_training_file.replace(".csv", ".shp"))

    uganda_data = pd.read_csv(uganda_training_file)
    uganda_data["geometry"] = uganda_data.apply(
        lambda row: Point(row["GPS (Longitude)"], row["GPS (Latitude)"]), axis=1
    )
    uganda_geo_data = gpd.GeoDataFrame(uganda_data, crs="EPSG:4326")
    uganda_geo_data.to_file(uganda_training_file.replace(".csv", ".shp"))


if __name__ == "__main__":
    main()
