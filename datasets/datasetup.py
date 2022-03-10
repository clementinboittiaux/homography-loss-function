import argparse
import glob
import os
import zipfile
from collections import namedtuple

from torchvision.datasets.utils import download_and_extract_archive

datasets = {
    'Cambridge': [
        'GreatCourt', 'KingsCollege', 'OldHospital', 'ShopFacade', 'StMarysChurch', 'Street'
    ],
    '7-Scenes': [
        'chess', 'fire', 'heads', 'office', 'pumpkin', 'redkitchen', 'stairs'
    ]
}
Scene = namedtuple('Scene', ['url', 'dataset'])
scenes = {
    'GreatCourt': Scene(
        url='https://www.repository.cam.ac.uk/bitstream/handle/1810/251291/GreatCourt.zip',
        dataset='Cambridge'
    ),
    'KingsCollege': Scene(
        url='https://www.repository.cam.ac.uk/bitstream/handle/1810/251342/KingsCollege.zip',
        dataset='Cambridge'
    ),
    'OldHospital': Scene(
        url='https://www.repository.cam.ac.uk/bitstream/handle/1810/251340/OldHospital.zip',
        dataset='Cambridge'
    ),
    'ShopFacade': Scene(
        url='https://www.repository.cam.ac.uk/bitstream/handle/1810/251336/ShopFacade.zip',
        dataset='Cambridge'
    ),
    'StMarysChurch': Scene(
        url='https://www.repository.cam.ac.uk/bitstream/handle/1810/251294/StMarysChurch.zip',
        dataset='Cambridge'
    ),
    'Street': Scene(
        url='https://www.repository.cam.ac.uk/bitstream/handle/1810/251292/Street.zip',
        dataset='Cambridge'
    ),
    'chess': Scene(
        url='http://download.microsoft.com/download/2/8/5/28564B23-0828-408F-8631-23B1EFF1DAC8/chess.zip',
        dataset='7-Scenes'
    ),
    'fire': Scene(
        url='http://download.microsoft.com/download/2/8/5/28564B23-0828-408F-8631-23B1EFF1DAC8/fire.zip',
        dataset='7-Scenes'
    ),
    'heads': Scene(
        url='http://download.microsoft.com/download/2/8/5/28564B23-0828-408F-8631-23B1EFF1DAC8/heads.zip',
        dataset='7-Scenes'
    ),
    'office': Scene(
        url='http://download.microsoft.com/download/2/8/5/28564B23-0828-408F-8631-23B1EFF1DAC8/office.zip',
        dataset='7-Scenes'
    ),
    'pumpkin': Scene(
        url='http://download.microsoft.com/download/2/8/5/28564B23-0828-408F-8631-23B1EFF1DAC8/pumpkin.zip',
        dataset='7-Scenes'
    ),
    'redkitchen': Scene(
        url='http://download.microsoft.com/download/2/8/5/28564B23-0828-408F-8631-23B1EFF1DAC8/redkitchen.zip',
        dataset='7-Scenes'
    ),
    'stairs': Scene(
        url='http://download.microsoft.com/download/2/8/5/28564B23-0828-408F-8631-23B1EFF1DAC8/stairs.zip',
        dataset='7-Scenes'
    )
}


def setup_scene(scene_str):
    scene = scenes[scene_str]
    download_and_extract_archive(scene.url, scene.dataset)
    os.remove(os.path.join(scene.dataset, scene.url.split('/')[-1]))
    if scene_str in datasets['7-Scenes']:
        for file in glob.glob(os.path.join(scene.dataset, scene_str, '*.zip')):
            with zipfile.ZipFile(file, 'r') as f:
                members = [m for m in f.namelist() if os.path.basename(m) != 'Thumbs.db']
                f.extractall(os.path.join(scene.dataset, scene_str), members=members)
            os.remove(file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        'dataset',
        choices=list(datasets.keys()) + list(scenes.keys()),
        help='name of the dataset or single scene to setup'
    )
    args = parser.parse_args()

    if args.dataset in datasets:
        for scene_name in datasets[args.dataset]:
            setup_scene(scene_name)
    else:
        setup_scene(args.dataset)
