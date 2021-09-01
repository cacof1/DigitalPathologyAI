# DigitalPathologyAI

First commit of the code for the UCL-led collaboration on Digital Pathology
In this code, we use conda and pytorch to uniformize effort.
To set up your Python environment:
- Install `conda` or `miniconda` for your operating system.

For Linux:
- Create a Conda environment from the `environment.yml` file in the repository root, and activate it:
```shell script
conda env create --file environment.yml
conda activate DigitalPathologyAI
```
For Windows:
- Create a Conda environment from the `environment_win.yml` file in the repository root, and activate it:
```shell script
conda env create --file environment_win.yml
conda activate DigitalPathologyAI
```

To then set up your file as a module, please use
```shell script
conda-develop /path/to/DigitalPathologyAI/
```

The first step to run the different code is to run the TileDataset.sh shell file, and to point it at your directory containing your datafiles
```shell script
source TileDataset.sh SlideDirectory/
```

This will create a folder (Preprocessing) containing mask, patches, and stitches for each slides with the coords of each tile of interest within each slide.