# OSM roads data 

The OSM roads data is derived from the base OSM data that exists at a country level - this raw data can be found in the Google Drive. 

More information about this specific OSM data type can be found [here](https://wiki.openstreetmap.org/wiki/Key:highway).

## Raw data 

Once the original zip file has been inflated, there should be a large number of shapefiles with the following paths:

`gis_osm_<data_type>_free_1.shp`

Unfortunately we cannot immediately rasterize these vector datasets as there is no numerical value to differentiate roads types (i.e. highway vs residential road). Therefore we have scripts to convert road types to integer values.

## Processing data

To run this script, it assumes that the road dataset is in the following path:

`data/osm/{country}/roads/gis_osm_roads_free_1.shp`

Once this is true run:

```
python src/scripts/get_value_from_osm_tag.py --source roads --country <country>
```

This will write a new dataset to:
`data/osm/{country}/roads/gis_osm_roads_free_1_with_tag_value.shp`

Now we can rasterize this vector dataset, using QGIS or GDAL. 

### GQIS rasterize (recommended)
Instructions for rasterizing using QGIS can be gound in the project's root README. Be sure to select this "layer" when converting.

### Gdal rasterize

The command from GDAL is:

```
gdal_rasterize data/osm/{country}/roads/gis_osm_roads_free_1_with_tag_value.shp data/osm/{country}/roads/{country}-roads.tif -tr 0.00027778 0.00027778 -a fclass_val
```

Which will write a tif file with 1 arc second resolution to:
`data/osm/{country}/roads/{country}-roads.tif`.

To change the resolution, simply adapt the values after the `-tr` argument in the command above.