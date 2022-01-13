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

#### Steps to Rasterize the data from GADM Data is as follows:
•	Open https://gadm.org/data.html for GADM Countrywide shapefiles<br/>
•	Click on Download Data by Country<br/>
•  Select Country from the drop-down menu<br/>
•	Download shapefiles in to a specific folder. Since we have level 4 for Uganda, we will be using level4 shape file for Rwanda.<br/>
•	Extract the data from zipped folder.<br/>

#### Converting Polygon to Line in QGIS:
•	Open QGIS Application<br/>
•	Add “gadm36_RWA_4.shp” file to layers section in QGIS Application. This is level four shape file.<br/>
•	Since the shape file is of polygons<br/>
•	We must convert this to line for creating raster.<br/>
•	To convert polygon to lines in QGIS<br/>
   o	Go to Vector tab<br/>
   o	Select Geometry Tools<br/>
   o	Select Polygon to Lines<br/>
•	You will get a dialog box to execute the conversion. Do not change anything and run the command.<br/>
•	After execution, we will get a temporary file(Layer) added to the layers section.<br/>
#### Raster Creation: 
•	To create a raster, go to “Raster” tab --> Conversion --> Rasterize (Vector to Raster)<br/>
•	Update parameters for raster and run the program.<br/>
•	You will get the raster file added to the Layers section and tiff file added to specified folder.<br/>
•	Follow the same process for Uganda shape files as well.<br/>
#### Parameters for Raster Creation:
•	Resolution: 1/3600 : 0.0002777 : This is the fixed value to burn in Raster Parameters.<br/>
•	Output Raster Size Units: Georeferenced Units<br/>
•	Width/ horizontal Resolution: 0.00030 for 30 mts width<br/>
•	Height/ vertical resolution: 0.00030 for 30 mts height<br/>
•	Output Extent: Calculate from Layer and select the line file we have created<br/>
•	Rasterized: Save data to a specific folder with required name.<br/>

#### Issues while processing Data:
•	Is we are not converting polygon shapes to lines, we will not be able to create rasters based on boundaries<br/>
•	Selection of Zoom levels is important while creating rasters<br/>
•	Selecting particular location for storing data is important or else we will get only temp file<br/>
•	Loading shape file from root folder is important or else we will not be able to create a layer in QGIS<br/>





