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
