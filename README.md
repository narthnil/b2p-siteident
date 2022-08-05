# B2P Remote Trailbridge Needs Assessment

## Install Anaconda environment and GDAL (MacOS, Ubuntu)

### Install Anaconda environment

```
conda create --name=b2p python=3.7
conda activate b2p
```

### OS X

[Gdal](https://gdal.org/) is a great tool for fast manipulation of rasters. However, it can be tricky to install. The best option it to install with Homebrew (in the current conda environment):

```
brew install gdal
```

Homebrew can be easily installed by following the instructions on [Homebrew's official page](https://brew.sh/).

Currently gdal is being used to rasterize vector data, using `gdal_rasterize` from the command line. Example:
```
gdal_rasterize data/osm/rwanda/roads/gis_osm_roads_free_1_with_tag_value.shp data/osm/rwanda/roads/gis_osm_roads_free_1_with_tag_value.tif -tr 0.0005 0.0005 -a fclass_val
```
The documentation on this service is found [here](https://gdal.org/programs/gdal_rasterize.html).

### Ubuntu

[Gdal](https://gdal.org/) can be also installed with Anaconda:

```
conda activate b2p
conda config --add channels conda-forge
conda config --set channel_priority strict
conda install gdal
```

## Install a anaconda environment and required packages
For reproducibility of all scripts, we can list all required packages in `requirements.txt`.
Please feel free to update the requirements.txt if you are using additional packages.

```
conda activate b2p

# choose one of the two installation
# 1) with cuda
pip install torch==1.10.0+cu113 torchvision==0.11.1+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
# 2) without cuda
pip install torch==1.10.0+cpu torchvision==0.11.1+cpu -f https://download.pytorch.org/whl/cpu/torch_stable.html

pip install -r requirements.txt

# optional
jupyter notebook --generate-config
```

# Preprocess data

Download `data.zip` from B2P Google drive and put the file into the root of this 
repository [(Link to B2P, permission is required)](https://drive.google.com/drive/folders/1sbJ8xUDyGOtcmO25q7ZQwPw3uxa3wHkF?usp=sharing).
Use the following commands to extract the files, move them to the `./data/` 
folder and calculate the statistics for training.

```
bash src/scripts/extract_data.sh
```

We also provide `processed_ground_truth.zip` in the same Google drive folder 
(link provided  above).If you download and extract it to `data/ground_truth`:
```
unzip processed_ground_truth.zip -d data/ground_truth/
rm processed_ground_truth.zip
```

You will work  with the same train/val/test split as we did for the evaluation. 
The folder structure would look like this:

```
data/ground_truth/
├── bounds_v1.geojson
├── bounds_v1_ssl.geojson
├── bounds_v2.geojson
├── bounds_v2_ssl.geojson
├── processed_ground_truth.zip
├── README.md
├── Rwanda training data_AllSitesMinusVehicleBridges_21.11.05.csv
├── stats.json
├── train_1200_v1.geojson
├── train_1200_v1_ssl.geojson
├── train_1200_v2.geojson
├── train_1200_v2_ssl.geojson
├── train_300_v1.geojson
├── train_300_v1_ssl.geojson
├── train_300_v2.geojson
├── train_300_v2_ssl.geojson
├── train_600_v1.geojson
├── train_600_v1_ssl.geojson
├── train_600_v2.geojson
├── train_600_v2_ssl.geojson
└── Uganda_TrainingData_3districts_ADSK.csv
```

If you want to create a random new train/val/test split, please execute the
following scripts:
```
python preprocess_train_data_v1.py
python preprocess_train_data_v2.py
python calculate_stats.py
```

## Manual handling of train data

Change in `data/ground_truth/Rwanda training data_AllSitesMinusVehicleBridges_21.11.05.csv` the following rows:
(line 864)
```
006f100000d7JDC,Rwanda - Gikomero - 1013421,2.442831,29.49934667
006f100000a86FN,Rwanda - Nyarurambi - 1007485,2.4724,29.5
006f100000a86GS,Rwanda - Coko - 1007552,2.713945,29.594535
```

to

```
006f100000d7JDC,Rwanda - Gikomero - 1013421,-2.442831,29.49934667
006f100000a86FN,Rwanda - Nyarurambi - 1007485,-2.4724,29.5
006f100000a86GS,Rwanda - Coko - 1007552,-2.713945,29.594535
```

Remove these rows in `data/ground_truth/Uganda_TrainingData_3districts_ADSK.csv`:

```
1023076,0063Z00000ixAgR,34.466806,-82.436215
1023313,0063Z00000iyr0b,,
1024477,0063Z00000kcJF1,,
1024494,0063Z00000kcREr,,
```

# Model training
All training parameters can be found in `src/argparser.py`. By default, it is required to specify a unique `--save_dir` 
(the directory to which training artifacts are saved). It is recommended to use the `results` base directory as those files will be ignored by git.

Supervised training:

```
python train_dist.py --model resnet50 --tile_size 1200 --save_dir results/resnet50-1200
```

Semi-supervised training:

```
python train_ssl.py --model resnet18 --tile_size 300 --out results/ssl-resnet50-300
```

## Rasterizing Vector Geometry using QGIS

The following are the instructions for rasterizing vector geometries using QGIS. 

1. Import the vector geometry into QGIS.
![](docs_imgs/AddVectorLayer.png)
This will present a new page which allows you to search for a specific file, after which you may select `Add`
2. Convert the vector geometry to raster.
![](docs_imgs/Rasterization.png)
3. Input parameters according to your specifications.
![](docs_imgs/RasterizationParameters.png)
The image above shows some of the parameters used previously, specifically:
- Input layer: Which file you want to convert
- Field to use for a burn-in value: Which feature in the vector geometry whose values will become the pixel values in the corresponding raster
- Output raster size units: Select `Georeferenced units` as this will ensure the raster is georeferenced
- Width/Height fields: We used one arc/second (1/3600) for the resolution
- Output extent: Select the three dots at the right edge and select for the output extent to be derived from the data
- Rasterized: Input the file destination of the raster (otherwise a temporary file will be used)
