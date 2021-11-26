import os

import pandas as pd
import torch
import tqdm
from PIL import Image
from torch.nn.functional import normalize
from torch.utils.data import Dataset
from torchvision import transforms

from quaternions import quaternion_to_R


def collate_fn(views):
    """
    Transforms list of dicts [{key1: value1, key2:value2}, {key1: value3, key2:value4}]
    into a dict of lists {key1: [value1, value3], key2: [value2, value4]}.
    Then stacks batch-compatible values into tensor batchs.
    """
    batch = {key: [] for key in views[0].keys()}
    for view in views:
        for key, value in view.items():
            batch[key].append(value)
    for key, value in batch.items():
        if key not in ['w_P', 'c_p', 'image_file']:
            batch[key] = torch.stack(value)
    return batch


class RelocDataset(Dataset):
    """
    Dataset template class for use with PyTorch DataLoader class.
    """
    def __init__(self, dataset):
        """
        `dataset` must be a list of dicts providing localization data for each image.
        Dicts must provide:
        {
            'image_file': name of image file
            'image': torch.tensor image with shape (3, height, width)
            'w_t_c': torch.tensor camera-to-world translation with shape (3, 1)
            'c_q_w': torch.tensor world-to-camera quaternion rotation with shape (4,) in format wxyz
            'c_R_w': torch.tensor world-to-camera rotation matrix with shape (3, 3)
                     (can be computed with quaternion_to_R)
            'K': torch.tensor camera intrinsics matrix with shape (3, 3)
            'w_P': torch.tensor 3D observations of the image in the world frame with shape (*, 3)
            'c_p': reprojections of the 3D observations in the camera view (in pixels) with shape (*, 3)
            'xmin': minimum depth of observations
            'xmax': maximum depth of observations
        }
        """
        self.data = dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {
            'image_file': self.data[idx]['image_file'],
            'image': self.data[idx]['image'],
            'w_t_c': self.data[idx]['w_t_c'],
            'c_q_w': self.data[idx]['c_q_w'],
            'c_R_w': self.data[idx]['c_R_w'],
            'K': self.data[idx]['K'],
            'w_P': self.data[idx]['w_P'],
            'c_p': self.data[idx]['c_p'],
            'xmin': self.data[idx]['xmin'],
            'xmax': self.data[idx]['xmax']
        }


class CambridgeDataset:
    """
    Template class to load every scene of Cambridge dataset.
    """
    def __init__(self, path):
        """
        `path` is the path to the dataset directory,
        e.g. for King's College: "/home/data/KingsCollege".
        Creates 6 attributes:
          - 2 lists of dicts (train and test) providing localization data for each image.
          - 4 parameters (train and test) for minimum and maximum depths of observations.
        """
        # Image preprocessing pipeline according to PyTorch implementation
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        views = []
        scene_coordinates = []
        with open(os.path.join(path, 'reconstruction.nvm'), mode='r') as file:

            # Skip first two lines
            for _ in range(2):
                file.readline()

            # `n_views` is the number of images
            n_views = int(file.readline())

            # For each image, NVM format is:
            # <File name> <focal length> <quaternion WXYZ> <camera center> <radial distortion> 0
            for _ in range(n_views):
                line = file.readline().split()

                f = float(line[1])
                K = torch.tensor([
                    [f, 0, 1920 / 2],
                    [0, f, 1080 / 2],
                    [0, 0, 1]
                ], dtype=torch.float32)
                views.append({
                    'image_file': line[0],
                    'K': K,
                    'observations_ids': []
                })

            # Skip one line
            file.readline()

            # `n_points` is the number of scene coordinates
            n_points = int(file.readline())

            # For each scene coordinate, SVM format is:
            # <XYZ> <RGB> <number of measurements> <List of Measurements>
            for i in range(n_points):

                line = file.readline().split()

                scene_coordinates.append(torch.tensor(list(map(float, line[:3]))))

                # `n_obs` is the number of images where the scene coordinate is observed
                n_obs = int(line[6])

                # Each measurement is
                # <Image index> <Feature Index> <xy>
                for n in range(n_obs):
                    views[int(line[7 + n * 4])]['observations_ids'].append(i)

        views = {view.pop('image_file'): view for view in views}
        scene_coordinates = torch.stack(scene_coordinates)

        train_df = pd.read_csv(os.path.join(path, 'dataset_train.txt'), sep=' ', skiprows=1)
        test_df = pd.read_csv(os.path.join(path, 'dataset_test.txt'), sep=' ', skiprows=1)

        train_data = []
        test_data = []
        train_global_depths = []
        test_global_depths = []

        print('Loading images from dataset. This may take a while...')
        for data, df, global_depths in [(train_data, train_df, train_global_depths),
                                        (test_data, test_df, test_global_depths)]:
            for line in tqdm.tqdm(df.values):
                image_file = line[0]
                image = preprocess(Image.open(os.path.join(path, image_file)))
                w_t_c = torch.tensor(line[1:4].tolist()).view(3, 1)
                c_q_w = normalize(torch.tensor(line[4:8].tolist()), dim=0)
                c_R_w = quaternion_to_R(c_q_w)[0]
                view = views[os.path.splitext(image_file)[0] + '.jpg']
                w_P = scene_coordinates[view['observations_ids']]
                c_P = c_R_w @ (w_P.T - w_t_c)
                c_p = view['K'] @ c_P
                c_p = c_p[:2] / c_p[2]

                args_inliers = torch.where(torch.logical_and(
                    torch.logical_and(
                        torch.logical_and(c_P[2] > 0.2, c_P[2] < 1000),
                        torch.logical_and(c_P[0].abs() < 1000, c_P[1].abs() < 1000)
                    ),
                    torch.logical_and(
                        torch.logical_and(c_p[0] > 0, c_p[0] < 1920),
                        torch.logical_and(c_p[1] > 0, c_p[1] < 1080)
                    )
                ))[0]

                if args_inliers.shape[0] < 10:
                    tqdm.tqdm.write(f'Not using image {image_file}: [{args_inliers.shape[0]}/{w_P.shape[0]}] scene '
                                    f'coordinates inliers')
                elif w_t_c.abs().max() > 1000:
                    tqdm.tqdm.write(f'Not using image {image_file}: t is {w_t_c.numpy()}')
                else:
                    if args_inliers.shape[0] != w_P.shape[0]:
                        tqdm.tqdm.write(f'Eliminating outliers in image {image_file}: '
                                        f'[{args_inliers.shape[0]}/{w_P.shape[0]}] scene coordinates inliers')

                    depths = torch.sort(c_P.T[args_inliers][:, 2]).values
                    global_depths.append(depths)

                    data.append({
                        'image_file': image_file,
                        'image': image,
                        'w_t_c': w_t_c,
                        'c_q_w': c_q_w,
                        'c_R_w': c_R_w,
                        'w_P': w_P[args_inliers],
                        'c_p': c_p.T[args_inliers],
                        'K': view['K'],
                        'xmin': depths[int(0.025 * (depths.shape[0] - 1))],
                        'xmax': depths[int(0.975 * (depths.shape[0] - 1))]
                    })

        train_global_depths = torch.sort(torch.hstack(train_global_depths)).values
        test_global_depths = torch.sort(torch.hstack(test_global_depths)).values
        self.train_global_xmin = train_global_depths[int(0.025 * (train_global_depths.shape[0] - 1))]
        self.train_global_xmax = train_global_depths[int(0.975 * (train_global_depths.shape[0] - 1))]
        self.test_global_dmin = test_global_depths[int(0.025 * (test_global_depths.shape[0] - 1))]
        self.test_global_dmax = test_global_depths[int(0.975 * (test_global_depths.shape[0] - 1))]
        self.train_data = train_data
        self.test_data = test_data
