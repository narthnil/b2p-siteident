// Import the dataset and select the elevation band.
var dataset = ee.Image('NASA/NASADEM_HGT/001');

// Load a country border as a region of interest (roi).
var countries = ee.FeatureCollection('USDOS/LSIB_SIMPLE/2017');

// Displaying slope for the region of interest.
var visualization_params = { min: 0, max: 90, palette: 'white, red' };


var slope = ee.Terrain.slope(dataset).select('slope');
var roi = countries.filterMetadata('country_na', 'equals', 'Uganda');
// Clip the image to the region of interest.
slope = slope.clip(roi);
Map.centerObject(roi);
Map.addLayer(slope, visualization_params, "slope");

// Export the image, specifying scale and region.

Export.image.toDrive({
    image: slope.visualize(visualization_params),
    region: roi,
    description: 'slope_uganda',
    scale: 30,
    maxPixels: 1e13
});

var slope = ee.Terrain.slope(dataset).select('slope');
var roi = countries.filterMetadata('country_na', 'equals', 'Rwanda');
// Clip the image to the region of interest.
slope = slope.clip(roi);
Map.centerObject(roi);
Map.addLayer(slope, visualization_params, "slope");

// Export the image, specifying scale and region.

Export.image.toDrive({
    image: slope.visualize(visualization_params),
    region: roi,
    description: 'slope_rwanda',
    scale: 30,
    maxPixels: 1e13
});
