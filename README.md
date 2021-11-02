# DigitalPathologyAI

## Installation
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

## Preprocessing
The first step to run the different code is to run the TileDataset.sh shell file, and to point it at your directory containing your datafiles
```shell script
source TileDataset.sh SlideDirectory/
```

This will, in order, do:

1. create a folder (Preprocessing) containing mask, patches, wsi, and stitches 
   1. A folder containing, for each slide, jpg showing the mask separation between tiles and background (mask)
   2. A folder containing, for each slide, h5 files containgCoords of each tile of interest within each slide (patches)
   3. A folder containing, for each slide, jpg showing a stitches of individual tiles at low resolution (stitches)
   4. A folder containing all of the wsi slides (moved there to minimize space usage) (wsi)
2. Apply a NN to extract the coords of the patches that are within the tumour and update the patches csv file
	 
The data processing workflow is displayed below, where the tumorous parts in the whole slides images are used as region of interest for developing multiple applications including tumour subtype classification, mitosis detection and unsupervised culstering.  	 
![image](https://user-images.githubusercontent.com/44832648/137453431-ebe11082-40f9-4b23-937e-41a78a5949e1.png)

## Code Division
The code has a series of separate goals to fulfill, and thus a modular programming approach is favoured. Preprocessing is shared by all codes and done with the shell file TileDataset.sh. Thereafter, the following blocks are suggested:
* **DataLoader** : Dataloading should be done using the DataLoader class, which is in the DataLoader subfolder. The class contains both a Data Generator (for inference), and a pytorch Data Module (for training). Important are the keywords arguments dim (dimension of the patch), vis_level (magnification level), and inference (boolean).
* **Model**: Models are stored in the Model subfolder. Generally, pre-written code should be imported from pytorch to minimize code error. Modified generic model are stored in small capital letter when needed (e.g. unet, resnet, where the latter are used for autoencoding). Code in Capital letters (e.g. AutoEncoding) will contains a full pytorch modular construction, and answer to a specific problem. These models may have a specific __main__ examplar, but should generally be used and defined as a standalone. 
* **Losses** : Subfolder containing non-generic loss functions
* **Applications** : Subfolder containing the assembled application, e.g. clustering, autoencoding, classifying. Each application will 1. load the DataLoader and data, 2. load the model, and perform training in batches and return the results. It might also perform inference where needed.
* **wsi_core** : Extracted snippet from the CLAM project that extract patches contours and remove background.


## Visualisation
Currently, the class Visualize can provide two funtions,
1. Creating the heatmap for visualizing the tumorous and normal areas;
2. Investigating and visualizing the feature vectors by dimension reduction approaches.

Besides, the FeatureEmbedding_vis_app.py can visaulize the 3D dimension reduction results of different patients using a web app.
