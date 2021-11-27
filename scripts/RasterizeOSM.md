# Rasterizing OSM data

Rasterizing OSM data is a two step process involving:
1. Converting the `tag` feature in OSM to a numerical value
2. Using `GDAL` to rasterize a vector dataset

## 1) Converting the tag feature in OSM to a numerical value

`get_value_from_osm_data.py` includes a script that accomplishes this.

To execute first make sure that your data is in the following file structure:
`data/osm/{country}/{source}/{file}.shp`
where
- `country` is the country the vector data refers to
- `source` is the type of OSM data (currently supports `waterways` and `roads`)
- `file` are the combined files that compose a shapefile 

You may then execute the script with:

```
python scripts/get_value_from_osm_tag.py --source <source> --country <country>
```

## 2 Using `GDAL` to rasterize a vector dataset

Refer to the root README for instructions on how to install GDAL. Then to rasterize the data run:

```
gdal_rasterize data/osm/{country}/{source}/{file}.shp data/osm/{country}/{source}/{file}.tif -tr 0.0005 0.0005 -a fclass_va
```