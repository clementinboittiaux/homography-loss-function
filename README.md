# Homography-based loss function for camera pose regression
In this repository, we share our implementation of several camera pose regression
loss functions in a simple end-to-end network similar to
[PoseNet](https://openaccess.thecvf.com/content_iccv_2015/html/Kendall_PoseNet_A_Convolutional_ICCV_2015_paper.html).
We implemented our homography-based loss functions and re-implemented PoseNet, Homoscedastic, Geometric and DSAC loss
functions. We provide the code to train the network and evaluate their performance on the Cambridge dataset.

<video width="320" height="240" controls>
    <source src="assets/animation.mp4" type="video/mp4">
</video>

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
The script [main.py](main.py) trains the network on a given scene.
It requires one positional argument: the path to the scene on which to train the model.
For example, for training the model on ShopFacade, simply run:
```bash
python main.py datasets/ShopFacade
```

Other available training options can be listed by running `python main.py -h`.

## Monitor training and test results
Training and test metrics are saved in a `logs` directory. One can monitor them using tensorboard.
Simply run in a new terminal:
```bash
tensorboard --logdir logs
```

All estimated poses are also saved in a CSV file.
