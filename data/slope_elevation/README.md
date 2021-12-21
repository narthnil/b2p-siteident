# Extracting slope and elevation data from Google Earth Engine

## Signing up for Google Earth Engine

In order to extract the data, you must be registered to Google Earth Engine
with a Google account. Go to
[earthengine.google.com](https://earthengine.google.com/) to sign-in.
Approval usually takes several hours to a day.

## Using Google Earth Engine Code editor to extract slope and elevation

The scripts `data/slope_elevation/elevation_rwanda_uganda.js` and
`data/slope_elevation/elevation_rwanda_uganda.js` to extract elevation and
slope. These scripts are specifically written to extract data for Rwanda and
Uganda. However, by changing `Uganda` (lines 12)

```
var roi = countries.filterMetadata('country_na', 'equals', 'Uganda');
```

to any other country, this script can be used to extract data of any other
country.
The script extract elevation / slope of a specific country. The script
normalizes the elevation values ($\min=0, \max=9000$) to values in range of
[0, 255]. Similarly, the script normalizes the slope values
($\min=0, \max=90$) to values in range of [0, 255].

In order to download these tiff images to Google drive, follow these
instructions:

1. Go to [code.earthengine.google.com](https://code.earthengine.google.com/).
2. Copy & paste one of the `js` files mentioned above into the editor of the
code edit.
3. Click on white button `Run`.
4. Click on `Tasks` on the most right tab. You should see two unsubmitted
tasks. 
5. Click on the blue button `Run` next to the task name. Check the task and
change `Drive folder`, `Filename*` if necessary.
6. Confirm the task run by clicking on `Run` in the dialogue. This will start
the job. Depending on the size of the country, the task can take 10-30 minutes.
7. Repeat steps 5-7 for the second task.
8. You can close the browser tab, even though the task may not have finished.
Once the task is finished, the files can be founds in your Google Drive folder.