# OSM waterways data 

The OSM waterways data is derived from the base OSM data that exists at a country level - this raw data can be found in the Google Drive. 

More information about this specific OSM data type can be found [here](https://wiki.openstreetmap.org/wiki/Key:waterway).

## Raw data 

Once the original zip file has been inflated, there should be a large number of shapefiles with the following paths:

`gis_osm_<data_type>_free_1.shp`

Unfortunately we cannot immediately rasterize these vector datasets as there is no numerical value to differentiate waterways types (i.e. river vs. stream). Therefore we have scripts to convert waterway types to integer values.

## Processing data

To run this script, it assumes that the waterway d dataset is in the following path:

`data/osm/{country}/waterways/gis_osm_waterways_free_1.shp`

Once this is true run:

```
python src/scripts/get_value_from_osm_tag.py --source waterways --country <country>
```

This will write a new dataset to:
`data/osm/{country}/waterways/gis_osm_waterways_free_1_with_tag_value.shp`

Now we can rasterize this vector dataset, using QGIS or GDAL. 

### Gdal rasterize

The command from GDAL is:

```
gdal_rasterize data/osm/{country}/waterways/gis_osm_waterways_free_1_with_tag_value.shp data/osm/{country}/waterways/{country}-waterways.tif -tr 0.00027778 0.00027778 -a fclass_val
```

Which will write a tif file with 1 arc second resolution to:
`data/osm/{country}/waterways/{country}-waterways.tif`.

To change the resolution, simply adapt the values after the `-tr` argument in the command above.