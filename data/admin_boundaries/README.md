# Admin boundaries
## Data processing (Mohsin Nadaf)

### Issues
Issue 1) Balck Polygon instead of line

Geometry in the adminboudary is of Polygon and MultiPolygon type if we rasterize this shape file we will get block instead of line.
So to avoid this we need to convert Polygon and MultiPolygon geometry type to Linestring and MultiLineString geometry.

To Solve this issue we can use ArcGis or we can use python code to conver Polygon and MultiPolygon geometry type to Linestring and MultiLineString geometry

Issue 2) Different CRS

For the Rwanda admin boundary we are getting EPGS-3857 CRS we need to convert it to EPGS-4326 otherwise we will get a large extent and we will be not able to rasterize Rwanda admin boundary.

To solve the above issue we can use the to_crs() method.


### Step by Step Process
1) Use "Raster Vector Data without any tif file.ipynb" python note book
2) call function pts2raster_without_tiff_file_input(........) to rasterize shp file
   
   Syntax to use:
   
   pts2raster_without_tiff_file_input(InputShapeFileName, OutputTiffFileName, Resolution)
   
   InputShapeFileName : Name of the shape file (including absolute/relative path) which we want to rasterize
   
   OutputTiffFileName : Name of the output file (including absolute/relative path)
   
   Resolution : As per Need
   
   Example:
   
   pts2raster_without_tiff_file_input("Uganda_Parish.shp","Uganda_Parish.tif", 0.000277)
   
## Data processing (Sai Alluru)
