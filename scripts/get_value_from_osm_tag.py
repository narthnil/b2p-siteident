import geopandas as gpd
from enum import Enum
from dataclasses import dataclass


@dataclass
class DataSourceParameters:
    default: int
    tag_to_value_map: dict
    file: str


class TagMap(Enum):
    ROADS = DataSourceParameters(
        default=8,
        tag_to_value_map={
            "footway": 0,
            "path": 1,
            "residential": 2,
            "track": 3,
            "tertiary": 4,
            "secondary": 5,
            "primary": 6,
            "trunk": 7,
        },
        file="data/osm/rwanda/roads/gis_osm_roads_free_1.shp",
    )
    WATERWAYS = DataSourceParameters(
        default=-1,
        tag_to_value_map={"canal": 0, "drain": 1, "stream": 2, "river": 3},
        file="data/osm/rwanda/waterways/gis_osm_waterways_free_1.shp",
    )


def assign_tag_to_value(tag, class_map, default):
    class_keys = list(class_map.keys())
    for key in class_keys:
        if tag in key:
            return class_map[key]
    return default


def main(data_source):
    if data_source == "road":
        tag_map_choices = TagMap.ROADS.value
    elif data_source == "waterways":
        tag_map_choices = TagMap.WATERWAYS.value
    else:
        raise Exception(f"{data_source} is not supported.")
    data = gpd.read_file(tag_map_choices.file)
    data["fclass_val"] = data["fclass"].apply(
        lambda tag: assign_tag_to_value(
            tag, tag_map_choices.tag_to_value_map, tag_map_choices.default
        )
    )
    out_path = tag_map_choices.file.replace(".shp", "_with_tag_value.shp")
    data.to_file(out_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, required=True, help="OSM data source")

    args = parser.parse_args()

    main(args.source)
