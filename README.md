# Homography-based loss function for camera pose regression
In this repository, we share our implementation of homography-based loss
functions in an end-to-end pose regression network, similarly as
[PoseNet](https://openaccess.thecvf.com/content_iccv_2015/html/Kendall_PoseNet_A_Convolutional_ICCV_2015_paper.html).
We also re-implemented PoseNet, Homoscedastic, Geometric and DSAC loss
functions. We provide the code to evaluate their performance on the Cambridge dataset.

## Installation

### Dataset setup
Have a look at the [datasets](datasets) folder to setup the Cambridge dataset.

### Python environment setup
We use `python 3.9.7` and `pip 21.2.4`. Modules requirements are listed in [requirements.txt](requirements.txt).
An easy way to setup the python environment is to have [Anaconda](https://www.anaconda.com) installed.  
Setting up an anaconda environment:
```bash
conda create -n homographyloss python=3.9.7 pip=21.2.4
conda activate homographyloss
pip install -r requirements.txt
```

## Run relocalization
