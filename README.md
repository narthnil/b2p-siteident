# B2P Remote Trailbridge Needs Assessment

## Install a anaconda environment and required packages
For reproducibility of all scripts, we can list all required packages in `requirements.txt`.
Please feel free to update the requirements.txt if you are using additional packages.

```
conda create --name=b2p python=3.7
conda activate b2p
pip install -r requirements.txt
# choose one of the two installation
# with cuda
# pip install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio==0.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
# without cuda
# pip install torch==1.10.0+cpu torchvision==0.11.1+cpu torchaudio==0.10.0+cpu -f https://download.pytorch.org/whl/cpu/torch_stable.html
```

Download [git-lfs](https://github.com/git-lfs/git-lfs) for managing large files with Git.
Once downloaded and installed, set up Git LFS for your user account by running:

```
git lfs install
```


## Install gdal

[Gdal](https://gdal.org/) is a great tool for fast manipulation of rasters. However, it can be tricky to install. The best option it to install with Homebrew (in the current conda environment):

```
brew install gdal
```

Currently gdal is being used to rasterize vector data, using `gdal_rasterize` from the command line. Example:
```
gdal_rasterize data/osm/rwanda/roads/gis_osm_roads_free_1_with_tag_value.shp data/osm/rwanda/roads/gis_osm_roads_free_1_with_tag_value.tif -tr 0.0005 0.0005 -a fclass_val
```
The documentation on this service is found [here](https://gdal.org/programs/gdal_rasterize.html).

# Preprocess data

Download `data.zip` from B2P Google drive and put the file into this repository.
Use the following commands to extract the files and move them to the `./data/` folder.

```
bash extract_data.sh
python preprocess_train_data.py
```

## Model training

```
python -m torch.distributed.launch --master_addr="127.0.0.2" --master_port=8798 --nproc_per_node=4 train.py
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