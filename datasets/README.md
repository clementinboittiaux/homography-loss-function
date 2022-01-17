# Cambridge dataset

Here you can find instructions to setup the Cambridge dataset.

## Description

The dataset is composed of 6 outdoor scenes. For each image in each scene
we have access to its pose and its 3D observations in the reconstructed model.

## Setup

This folder contains 7 scripts. There is one script per scene for individual scene setup and one
script to setup all scenes at once.

### Requirements
Setup scripts require `wget` and `unzip` installed.

### Installation
For an individual scene setup, e.g. Shop Façade, simply run in a terminal:
```bash
./setup_shopfacade.sh
```

If you want to setup all scenes at once, simply run:
```bash
./setup_all_scenes.sh
```
