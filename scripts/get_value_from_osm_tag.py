import geopandas as gpd


default = 8
class_map = {
    "footway": 0,
    "path": 1,
    "residential": 2,
    "track": 3,
    "tertiary": 4,
    "secondary": 5,
    "primary": 6,
    "trunk": 7
}
class_keys = list(class_map.keys())

def assign_road_tag_to_value(tag):
    for key in class_keys:
        if tag in key:
            return class_map[key]
    return default


def main():
    road_data = gpd.read_file("data/osm/rwanda/roads/gis_osm_roads_free_1.shp")
    road_data["fclass_val"] = road_data["fclass"].apply(lambda tag: assign_road_tag_to_value(tag))
    road_data.to_file("data/osm/rwanda/roads/roads_with_value.shp")


if __name__ == "__main__":
    main()
