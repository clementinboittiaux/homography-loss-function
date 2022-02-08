import glob
import os

import numpy as np
import pandas as pd
import torch
import tqdm
from PIL import Image
from kornia.geometry.conversions import rotation_matrix_to_quaternion, QuaternionCoeffOrder
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
            'c_p': reprojections of the 3D observations in the camera view (in pixels) with shape (*, 2)
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

    def __init__(self, path, xmin_percentile, xmax_percentile):
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
                        'xmin': depths[int(xmin_percentile * (depths.shape[0] - 1))],
                        'xmax': depths[int(xmax_percentile * (depths.shape[0] - 1))]
                    })

        train_global_depths = torch.sort(torch.hstack(train_global_depths)).values
        test_global_depths = torch.sort(torch.hstack(test_global_depths)).values
        self.train_global_xmin = train_global_depths[int(xmin_percentile * (train_global_depths.shape[0] - 1))]
        self.train_global_xmax = train_global_depths[int(xmax_percentile * (train_global_depths.shape[0] - 1))]
        self.test_global_xmin = test_global_depths[int(xmin_percentile * (test_global_depths.shape[0] - 1))]
        self.test_global_xmax = test_global_depths[int(xmax_percentile * (test_global_depths.shape[0] - 1))]
        self.train_data = train_data
        self.test_data = test_data


class SevenScenesDataset:
    """
    Template class to load every scene from 7-Scenes dataset
    """

    def __init__(self, path, xmin_percentile, xmax_percentile):
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Camera intrinsics
        K = np.array([
            [585, 0, 320],
            [0, 585, 240],
            [0, 0, 1]
        ], dtype=np.float64)
        K_inv = np.linalg.inv(K)
        K_torch = torch.tensor(K, dtype=torch.float32)

        # Grid of pixels
        u = np.arange(640) + 0.5
        v = np.arange(480) + 0.5
        u, v = np.meshgrid(u, v)

        # Array of all pixel positions in pixels
        c_p_px = np.hstack([
            u.reshape(-1, 1),
            v.reshape(-1, 1),
            np.ones((u.size, 1))
        ])

        # Array of all pixels in the sensor plane
        c_p = K_inv @ c_p_px.T

        train_data = []
        test_data = []
        train_global_depths = []
        test_global_depths = []

        for data, file, global_depths in [(train_data, 'TrainSplit.txt', train_global_depths),
                                          (test_data, 'TestSplit.txt', test_global_depths)]:

            with open(os.path.join(path, file), mode='r') as f:
                seqs = [int(line[8:]) for line in f]

            for seq in seqs:

                seq_dir = os.path.join(path, f'seq-{seq:02d}')

                print(f'Loading seq-{seq:02d}')

                for frame in tqdm.tqdm(glob.glob(os.path.join(seq_dir, '*.color.png'))):

                    frame = os.path.basename(frame).split('.')[0]
                    image_path = os.path.join(seq_dir, f'{frame}.color.png')
                    pose_path = os.path.join(seq_dir, f'{frame}.pose.txt')
                    depth_path = os.path.join(seq_dir, f'{frame}.depth.png')

                    image = preprocess(Image.open(image_path))

                    # Read camera-to-world pose
                    w_M_c = np.zeros((4, 4))
                    with open(pose_path, mode='r') as f:
                        for i, line in enumerate(f):
                            w_M_c[i] = list(map(float, line.strip().split('\t')))

                    # Read depth map
                    Z = np.array(Image.open(depth_path)).reshape(-1, 1)

                    # Filter outliers
                    args_inliers = np.logical_and(Z > 0, Z != 65535).squeeze()

                    # Unproject pixels
                    c_P = c_p.T[args_inliers] * (Z[args_inliers] / 1000)

                    # Convert 3D points from camera to world frame
                    w_P = w_M_c[:3, :3] @ c_P.T + w_M_c[:3, 3:4]

                    # Building rotation matrix and its quaternion
                    w_M_c = torch.tensor(w_M_c)
                    c_R_w = w_M_c[:3, :3].T.contiguous()
                    c_q_w = rotation_matrix_to_quaternion(c_R_w, order=QuaternionCoeffOrder.WXYZ)

                    # Keep the quaternion on the top hypershpere
                    if c_q_w[0] < 0:
                        c_q_w *= -1

                    # Sort depths
                    depths = Z[args_inliers].flatten()
                    global_depths.append(depths)
                    depths = np.sort(depths)

                    data.append({
                        'image_file': f'seq-{seq:02d}/{frame}.color.png',
                        'image': image,
                        'w_t_c': w_M_c[:3, 3:4].float(),
                        'c_q_w': c_q_w.float(),
                        'c_R_w': c_R_w.float(),
                        'w_P': torch.tensor(w_P.T, dtype=torch.float32),
                        'c_p': torch.tensor(c_p_px[args_inliers, :2], dtype=torch.float32),
                        'K': K_torch,
                        'xmin': torch.tensor(
                            depths[int(xmin_percentile * (depths.size - 1))] / 1000, dtype=torch.float32
                        ),
                        'xmax': torch.tensor(
                            depths[int(xmax_percentile * (depths.size - 1))] / 1000, dtype=torch.float32
                        )
                    })

        # Sort global depths
        print('Sorting depths, this may take a while...')
        train_global_depths = np.sort(np.hstack(train_global_depths))
        test_global_depths = np.sort(np.hstack(test_global_depths))

        self.train_global_xmin = torch.tensor(
            train_global_depths[int(xmin_percentile * (train_global_depths.size - 1))] / 1000,
            dtype=torch.float32
        )
        self.train_global_xmax = torch.tensor(
            train_global_depths[int(xmax_percentile * (train_global_depths.size - 1))] / 1000,
            dtype=torch.float32
        )
        self.test_global_xmin = torch.tensor(
            test_global_depths[int(xmin_percentile * (test_global_depths.size - 1))] / 1000,
            dtype=torch.float32
        )
        self.test_global_xmax = torch.tensor(
            test_global_depths[int(xmax_percentile * (test_global_depths.size - 1))] / 1000,
            dtype=torch.float32
        )
        self.train_data = train_data
        self.test_data = test_data
