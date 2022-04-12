# Dataset setup

Here you can find instructions to setup datasets for use with this code.

## Cambridge and 7-Scenes

We provide the script [datasetup.py](datasetup.py) for setting up Cambridge and 7-Scenes datasets. The script can be
called with either the name of the dataset to setup, *e.g.*, `7-Scenes`, or the name of a specific scene, *e.g.*,
`KingsCollege`. For example, if you want to setup the whole Cambridge dataset:
```shell
python datasetup.py Cambridge
```
Or if you want to only setup the *chess* scene of 7-Scenes dataset:
```shell
python datasetup.py chess
```
All possibilities can be accessed by running:
```shell
python datasetup.py -h
```


## Custom dataset

We also support custom datasets in **COLMAP** model format.  
⚠️ Please note that only **RADIAL** camera models are supported for now.

The custom dataset folder must contain:
- The COLMAP model: `cameras`, `images` and `points3D` files in `.bin` or `.txt` format.
- A folder named `images` containing all images in the model.
- A file named `list_db.txt` with the name of all the images used for training, one image name per line.
- A file named `list_query.txt` with the name of all the images used for testing, one image name per line.

The final outline of the folder should look like this:
> - mydataset
>   - images
>     - frame001.jpg
>     - frame002.jpg
>     - frame003.jpg
>     - ...
>   - cameras.bin
>   - images.bin
>   - points3D.bin
>   - list_db.txt
>   - list_query.txt

An example of `list_db.txt` or `list_query.txt`:
```text
frame001.jpg
frame002.jpg
frame003.jpg
...
```
