# Extracting OSM map images

This extraction is done with QGIS. Please download QGIS at the
[official page](https://www.qgis.org/en/site/).

## Add XYZ tile server based layer
Go to `Layer -> Add Layer -> Add XYZ Layer...` and create a new layer for the
wmflabs OSM without labels under the connection url:

```
https://tiles.wmflabs.org/osm-no-labels/${z}/${x}/${y}.png	
```

taken from 
[OSM tile servers](https://wiki.openstreetmap.org/wiki/Tile_servers).

## Projection

Set the projection to `EPSG:4326`.

## Extract image

1. Go to `Project -> Import/Export -> Export Map to Image...`.
2. Set the scale to 1:50000.
3. Use the boundaries below for Rwanda and Uganda to fill in the extent.
4. Press `Save` to save the image to file.

## Boundaries

Rwanda:

* North: -1.054444551
* East: 30.893260956
* South: -2.825486183
* West: 28.854442596

Uganda:

* North: 4.222777367
* East: 35.009719849
* South: -1.476110339
* West: 29.574302673
