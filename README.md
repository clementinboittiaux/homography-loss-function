# Homography-based loss function for camera pose regression
We share the implementation of the homography-based loss
function in an end-to-end pose regression network. We also provide
the code to test its performance on the Cambridge dataset. In addition to
that, we re-implemented PoseNet, Homoscedastic, Geometric and DSAC loss
functions to compare their performance with ours.

We use `python 3.9.7` and `pip 21.2.4`.
For example when setting up a conda environment :
```bash
conda create -n homographyloss python=3.9.7 pip=21.2.4
conda activate homographyloss
pip install -r requirements.txt
```
