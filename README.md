# B2P Remote Trailbridge Needs Assessment

## Install a anaconda environment and required packages
For reproducibility of all scripts, we can list all required packages in `requirements.txt`.
Please feel free to update the requirements.txt if you are using additional packages.

```
conda create --name=b2p python=3.7
conda activate b2p
pip install -r requirements.txt
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
gdal_rasterize data/osm/rwanda/roads/roads_with_value.shp data/osm/rwanda/roads/osm_roads.tif -tr 0.0005 0.0005 -a fclass_val
```
The documentation on this service is found [here](https://gdal.org/programs/gdal_rasterize.html).