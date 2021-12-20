# Admin boundaries
## Data processing

## Issues
Issue 1) Balck Polygon instead of line

Geometry in adminboudary is of Polygon and MultiPolygon type if we rasterize this shp file we will get block instead of line.
So to avoid this we need to convert Polygon and MultiPolygon geometry type to Linestring and MultiLineString geometry.

To Solve above above issue we can use ArcGis or we can use python code to conver Polygon and MultiPolygon geometry type to Linestring and MultiLineString geometry

Issue 2) Different CRS

For Rwanda admin boundary we are getting EPGS-3857 CRS we need to convert it to EPGS-4326 otherwise we will get large extent and we will be not able to rasterize Rwanda admin boundary.

To solve above issue we can use to_crs() method.

## Step by Step Process
1) Use "Raster Vector Data without any tif file.ipynb" python note book
2) call function pts2raster_without_tiff_file_input(........) to rasterize shp file
   
   Syntax to use:
   pts2raster_without_tiff_file_input(InputShapeFileName, OutputTiffFileName, Resolution)
   
   Example:
   pts2raster_without_tiff_file_input("Uganda_Parish.shp","Uganda_Parish.tif", 0.000277)
