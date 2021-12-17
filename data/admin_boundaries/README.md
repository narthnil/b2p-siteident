# Admin boundaries
## Data processing

Issue 1) Balck Polygon instead of line

Geometry in adminboudary is of Polygon and MultiPolygon type if we rasterize this shp file we will get block instead of line.
So to avoid this we need to convert Polygon and MultiPolygon geometry type to Linestring and MultiLineString geometry.

To Solve above above issue we can use ArcGis or we can use python code to conver Polygon and MultiPolygon geometry type to Linestring and MultiLineString geometry

Issue 2) Different CRS

For Rwanda admin boundary we are getting EPGS-3857 CRS we need to convert it to EPGS-4326 otherwise we will get large extent and we will be not able to rasterize Rwanda admin boundary.
 
To solve above issue we can use to_crs() method.

After resolving above two issue we can use QGIS or ArcGis to Rasterize shp Files
