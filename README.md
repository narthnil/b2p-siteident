# B2P Remote Trailbridge Needs Assessment

## Install GDAL (OS X, Ubuntu)

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

jupyter notebook --generate-config
```

# Preprocess data

Download `data.zip` from B2P Google drive and put the file into this repository [(Link to B2P, permission is required)](https://drive.google.com/drive/folders/1sbJ8xUDyGOtcmO25q7ZQwPw3uxa3wHkF?usp=sharing).
Use the following commands to extract the files, move them to the `./data/` folder and calculate the statistics for training.

```
bash extract_data.sh
python preprocess_train_data_v1.py
python preprocess_train_data_v2.py
python calculate_stats.py
```

## Model training

```
python train.py --model resnet50 --tile_size 1200 --save_dir results/resnet50-1200
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