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

For that click on `New`, add a `Name` and the URL from above.

In order to see this map, drag the map from the `Browser` panel (if you don't see the browser panel, then click `View -> Panels -> Browser`) to the `Layers` panel.

## Projection

Set the projection to `EPSG:4326`. The projection is shown in the bottom right next to the option `Render`.

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

Ethiopia:

* North: 14.883609772
* East: 47.988187290
* South: 3.406665328
* West: 32.991798401

## Data structure

The data is structured as followed after using the instructions:

```
data/osm/imgs/
├── README.md
├── rwanda_osm_nolab_1-50000_4326.tiff
├── uganda_osm_nolab_1-50000_4326.tiff
└── uganda_train_osm_nolab_1-50000_4326.tiff
```
