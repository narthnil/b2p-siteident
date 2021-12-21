// Import the dataset and select the elevation band.
var dataset = ee.Image('NASA/NASADEM_HGT/001');

// Load a country border as a region of interest (roi).
var countries = ee.FeatureCollection('USDOS/LSIB_SIMPLE/2017');

// Displaying slope for the region of interest.
var visualization_params = { min: 0, max: 9000, palette: 'white, red' };


var elevation = dataset.select('elevation');
var roi = countries.filterMetadata('country_na', 'equals', 'Uganda');
// Clip the image to the region of interest.
elevation = elevation.clip(roi);
Map.centerObject(roi);
Map.addLayer(elevation, visualization_params, "elevation");

// Export the image, specifying scale and region.

Export.image.toDrive({
  //image: elevation.visualize(visualization_params),
  image: elevation,
  region: roi,
  description: 'elevation_uganda_nonorm',
  scale: 30,
  maxPixels: 1e12
});

var elevation = dataset.select('elevation');
var roi = countries.filterMetadata('country_na', 'equals', 'Rwanda');
// Clip the image to the region of interest.
elevation = elevation.clip(roi);
Map.centerObject(roi);
Map.addLayer(elevation, visualization_params, "elevation");

// Export the image, specifying scale and region.

Export.image.toDrive({
  //image: elevation.visualize(visualization_params),
  image: elevation,
  region: roi,
  description: 'elevation_rwanda_nonorm',
  scale: 30,
  maxPixels: 1e12
});
