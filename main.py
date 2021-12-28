import argparse
import os
import tqdm
import random

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import datasets
import losses
import models
from utils import batch_to_device, batch_errors, batch_compute_utils, log_poses, log_errors

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('path', help='Full path to the dataset directory, e.g. "/home/data/KingsCollege"')
    parser.add_argument(
        '--loss', help='Loss function for training',
        choices=['local_homography', 'global_homography', 'posenet', 'homoscedastic', 'geometric', 'dsac'],
        default='local_homography'
    )
    parser.add_argument(
        '--device', choices=['cuda', 'cpu'],
        help='"cuda" for GPU computing, else "cpu"',
        default='cpu'
    )
    args = parser.parse_args()

    # Set seed for reproductibility
    seed = 1
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Load model
    model = models.load_model()
    model.train()
    model.to(args.device)

    # Load dataset
    dataset = datasets.CambridgeDataset(args.path)

    # Wrapper for use with PyTorch's DataLoader
    train_dataset = datasets.RelocDataset(dataset.train_data)
    test_dataset = datasets.RelocDataset(dataset.test_data)

    # Creating data loaders for train and test data
    train_loader = DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True,
        pin_memory=True,
        collate_fn=datasets.collate_fn,
        drop_last=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=64,
        shuffle=False,
        pin_memory=True,
        collate_fn=datasets.collate_fn
    )

    # Adam optimizer default epsilon parameter is 1e-8
    eps = 1e-8

    # Instantiate loss
    if args.loss == 'local_homography':
        criterion = losses.LocalHomographyLoss(device=args.device)
        eps = 1e-14  # Adam optimizer epsilon is set to 1e-14 for homography losses
    elif args.loss == 'global_homography':
        criterion = losses.GlobalHomographyLoss(
            xmin=dataset.train_global_xmin,
            xmax=dataset.train_global_xmax,
            device=args.device
        )
        eps = 1e-14  # Adam optimizer epsilon is set to 1e-14 for homography losses
    elif args.loss == 'posenet':
        criterion = losses.PoseNetLoss(beta=500)
    elif args.loss == 'homoscedastic':
        criterion = losses.HomoscedasticLoss(s_hat_t=0.0, s_hat_q=-3.0, device=args.device)
    elif args.loss == 'geometric':
        criterion = losses.GeometricLoss()
    elif args.loss == 'dsac':
        criterion = losses.DSACLoss()
    else:
        raise Exception(f'Loss {args.loss} not recognized...')

    # Instantiate adam optimizer
    optimizer = torch.optim.Adam(list(model.parameters()) + list(criterion.parameters()), lr=1e-4, eps=eps)

    # Set up tensorboard
    writer = SummaryWriter(os.path.join('logs', os.path.basename(os.path.normpath(args.path)), args.loss))

    # Set up folder to save weights
    if not os.path.exists(os.path.join(writer.log_dir, 'weights')):
        os.makedirs(os.path.join(writer.log_dir, 'weights'))

    # Set up file to save logs
    log_file_path = os.path.join(writer.log_dir, 'epochs_poses_log.csv')
    with open(log_file_path, mode='w') as log_file:
        log_file.write('epoch,image_file,type,w_tx_chat,w_ty_chat,w_tz_chat,chat_qw_w,chat_qx_w,chat_qy_w,chat_qz_w\n')

    print('Start training...')
    for epoch in tqdm.tqdm(range(5000)):
        epoch_loss = 0
        t_errors, q_errors, reprojection_errors = [], [], []

        for batch in train_loader:
            optimizer.zero_grad()

            # Move all batch data to proper device
            batch = batch_to_device(batch, args.device)

            # Estimate the pose from the image
            batch['w_t_chat'], batch['chat_q_w'] = model(batch['image']).split([3, 4], dim=1)

            # Computes useful data for our batch
            # - Normalized quaternion
            # - Rotation matrix from this normalized quaternion
            # - Reshapes translation component to fit shape (batch_size, 3, 1)
            batch_compute_utils(batch)

            # Compute loss
            loss = criterion(batch)

            # Backprop
            loss.backward()
            optimizer.step()

            # Add current batch loss to epoch loss
            epoch_loss += loss.item() / len(train_loader)

            # Compute training batch errors and log poses
            with torch.no_grad():
                batch_t_errors, batch_q_errors, batch_reprojection_errors = batch_errors(batch)
                t_errors.append(batch_t_errors)
                q_errors.append(batch_q_errors)
                reprojection_errors += batch_reprojection_errors

                with open(log_file_path, mode='a') as log_file:
                    log_poses(log_file, batch, epoch, 'train')

        # Log epoch loss
        writer.add_scalar('train loss', epoch_loss, epoch)

        with torch.no_grad():

            # Log train errors
            log_errors(t_errors, q_errors, reprojection_errors, writer, epoch, 'train')

            # Set the model to eval mode for test data
            model.eval()
            t_errors, q_errors, reprojection_errors = [], [], []

            for batch in test_loader:

                # Compute test poses estimations
                batch = batch_to_device(batch, args.device)
                batch['w_t_chat'], batch['chat_q_w'] = model(batch['image']).split([3, 4], dim=1)
                batch_compute_utils(batch)

                # Log test poses
                with open(log_file_path, mode='a') as log_file:
                    log_poses(log_file, batch, epoch, 'test')

                # Compute test errors
                batch_t_errors, batch_q_errors, batch_reprojection_errors = batch_errors(batch)
                t_errors.append(batch_t_errors)
                q_errors.append(batch_q_errors)
                reprojection_errors += batch_reprojection_errors

            # Log test errors
            log_errors(t_errors, q_errors, reprojection_errors, writer, epoch, 'test')

            # Log loss parameters, if there are any
            for p_name, p in criterion.named_parameters():
                writer.add_scalar(p_name, p, epoch)

            writer.flush()
            model.train()

            # Save model and optimizer weight every n epochs:
            if epoch % 10 == 0:
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()
                }, os.path.join(writer.log_dir, 'weights', f'epoch_{epoch}.pth'))

    writer.close()
