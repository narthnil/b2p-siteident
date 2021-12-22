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

Go to `Project -> Import/Export -> Export Map to Image...`. Set the scale to
1:50000. Use the boundaries below for Rwanda and Uganda to fill in the extent.
Press `Save` to save the image to file.

## Boundaries

Rwanda:

* North: -1.054444551
* East: 30.893260956
* South: -2.825486183
* West: 28.854442596

Uganda (not used):

* North: 4.222777367
* East: 35.009719849
* South: -1.476110339
* West: 29.574302673

B2P specific Uganda area:

* North: 0.9
* East: 30.7
* South: -0.28
* West: 29.6

##

```
import rasterio
from rasterio.merge import merge
from rasterio.plot import show

# open image parts of Uganda
src_pt1 = rasterio.open(
    "./data/osm/imgs/uganda_pt1_osm_nolab_1-50000_4326.tiff")
src_pt2 = rasterio.open(
    "./data/osm/imgs/uganda_pt2_osm_nolab_1-50000_4326.tiff")

# merge images
src, out_transf = merge([src_pt1, src_pt2])

# copy and adjust metadata
out_meta = src_pt1.meta.copy()
out_meta.update({
    "driver": "GTiff",
    "height": src.shape[1],
    "width": src.shape[2],
    "transform": out_transf
})

# write to file
with rasterio.open(
        "./data/osm/imgs/uganda_osm_nolab_1-50000_4326.tiff",
        "w", **out_meta) as f:
    f.write(src)


```