# Admin boundaries
## Data processing

Issue 1) Balck Polygon instead of line
Geometry in adminboudary is of Polygon and MultiPolygon type if we rasterize this shp file we will block instead of line.
So to avoid this we need to convert Polygon and MultiPolygon geometry type to linestring geometry.

Issue 2) Different CRS
For Rwanda we are getting EPGS--- CRS we need to convert it to EPGS--- otherwise we will get large extent and we will be not able to rasterize Rwanda admin boundary.

After resolving above two issue we can use  QGIS or ArcGis to Rasterize shp Files
