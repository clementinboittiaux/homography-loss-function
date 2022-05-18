import torch
from torch.nn.functional import normalize
from kornia.geometry.conversions import quaternion_to_rotation_matrix, QuaternionCoeffOrder


def angle_between_quaternions(q, r):
    """
    Works on batchs of quaternions only.
    `q` and `r` must be batchs of unit quaternions with shape (n, 4).
    """
    return 2 * torch.sum(q * r, dim=1).abs().clip(0, 1).arccos()


def l1_loss(input, target, reduce='mean'):
    """
    Computes batch L1 loss with `reduce` reduction.
    `input` and `target` must have shape (batch_size, *).
    L1 norm will be computed for each element on the batch.
    """
    loss = torch.abs(target - input).sum(dim=1)
    if reduce == 'none':
        return loss
    elif reduce == 'mean':
        return loss.mean()
    else:
        raise Exception(f'Reduction method {reduce} not known')


def l2_loss(input, target, reduce='mean'):
    """
    Computes batch L2 loss with `reduce` reduction.
    `input` and `target` must have shape (batch_size, *).
    L2 norm will be computed for each element on the batch.
    """
    loss = torch.square(target - input).sum(dim=1).sqrt()
    if reduce == 'none':
        return loss
    elif reduce == 'mean':
        return loss.mean()
    else:
        raise Exception(f'Reduction method {reduce} not known')


def compute_ABC(w_t_c, c_R_w, w_t_chat, chat_R_w, c_n, eye):
    """
    Computes A, B, and C matrix given estimated and ground truth poses
    and normal vector n.
    `w_t_c` and `w_t_chat` must have shape (batch_size, 3, 1).
    `c_R_w` and `chat_R_w` must have shape (batch_size, 3, 3).
    `n` must have shape (3, 1).
    `eye` is the (3, 3) identity matrix on the proper device.
    """
    chat_t_c = chat_R_w @ (w_t_c - w_t_chat)
    chat_R_c = chat_R_w @ c_R_w.transpose(1, 2)

    A = eye - chat_R_c
    C = c_n @ chat_t_c.transpose(1, 2)
    B = C @ A
    A = A @ A.transpose(1, 2)
    B = B + B.transpose(1, 2)
    C = C @ C.transpose(1, 2)

    return A, B, C


def batch_to_device(batch, device):
    """
    If `device` is not 'cpu', moves all data in batch to the GPU.
    """
    if device != 'cpu':
        for key, value in batch.items():
            if type(value) is torch.Tensor:
                batch[key] = value.to(device)
            elif type(value[0]) is torch.Tensor:
                for index_value, value_value in enumerate(value):
                    value[index_value] = value_value.to(device)
    return batch


def project(w_t_c, c_R_w, w_P, K=None):
    """
    Projects 3D points P expressed in frame w onto frame c camera view.
    `w_t_c` is the (3, 1) shaped translation from frame c to frame w.
    `c_R_w` is the (3, 3) rotation matrix from frame w to frame c.
    `w_P` are (n, 3) shaped 3D points P expressed in the w frame.
    `K` is frame c camera matrix.
    """
    c_p = c_R_w @ (w_P.T - w_t_c)
    if K is not None:
        c_p = K @ c_p
    c_p = c_p[:2] / c_p[2]
    return c_p


def batch_errors(batch, errors):
    """
    Computes translation, rotation and reprojection errors for the batch.
    """
    b_errors = {
        't_errors': [l2_loss(batch['w_t_chat'], batch['w_t_c'], reduce='none').squeeze()],
        'q_errors': [angle_between_quaternions(batch['normalized_chat_q_w'], batch['c_q_w'])],
        'reprojection_error_sum': 0,
        'reprojection_distance_sum': 0,
        'l1_reprojection_error_sum': 0,
        'n_points': 0
    }

    for w_t_chat, chat_R_w, w_P, c_p, K in zip(batch['w_t_chat'], batch['chat_R_w'], batch['w_P'],
                                               batch['c_p'], batch['K']):
        chat_p = project(w_t_chat, chat_R_w, w_P, K=K)
        diff = chat_p.T - c_p
        reprojection_errors = torch.square(diff).sum(dim=1).clip(0, 1000000)
        b_errors['reprojection_error_sum'] += reprojection_errors.sum()
        b_errors['reprojection_distance_sum'] += reprojection_errors.sqrt().sum()
        b_errors['l1_reprojection_error_sum'] += torch.abs(diff).sum(dim=1).clip(0, 1000000).sum()
        b_errors['n_points'] += c_p.shape[0]

    if len(errors) == 0:
        for key, value in b_errors.items():
            errors[key] = value
    else:
        for key, value in b_errors.items():
            errors[key] += value


def batch_compute_utils(batch):
    """
    Computes inplace useful data for the batch.
    - Computes a normalized quaternion, and its corresponding rotation matrix.
    - Reshapes translation component to fit shape (batchs_size, 3, 1).
    """
    batch['w_t_chat'] = batch['w_t_chat'].view(-1, 3, 1)
    batch['normalized_chat_q_w'] = normalize(batch['chat_q_w'], dim=1)
    batch['chat_R_w'] = quaternion_to_rotation_matrix(batch['chat_q_w'], order=QuaternionCoeffOrder.WXYZ)


def log_poses(log_file, batch, epoch, data_type):
    """
    Logs batch estimated poses in log file.
    """
    log_file.write('\n'.join([
        f'{epoch},{image_file},{data_type},{",".join(map(str, w_t_chat.squeeze().tolist()))},'
        f'{",".join(map(str, chat_q_w.tolist()))}'
        for image_file, w_t_chat, chat_q_w in
        zip(batch['image_file'], batch['w_t_chat'], batch['chat_q_w'])]) + '\n'
    )


def log_errors(errors, writer, epoch, data_type):
    """
    Logs epoch poses errors in tensorboard.
    """
    t_errors = torch.hstack(errors['t_errors'])
    q_errors = torch.hstack(errors['q_errors']).rad2deg()

    writer.add_scalar(f'{data_type} distance median', t_errors.median(), epoch)
    writer.add_scalar(f'{data_type} angle median', q_errors.median(), epoch)
    writer.add_scalar(
        f'{data_type} mean reprojection error',
        errors['reprojection_error_sum'] / errors['n_points'], epoch
    )
    writer.add_scalar(
        f'{data_type} mean reprojection distance',
        errors['reprojection_distance_sum'] / errors['n_points'], epoch
    )
    writer.add_scalar(
        f'{data_type} mean l1 reprojection error',
        errors['l1_reprojection_error_sum'] / errors['n_points'], epoch
    )

    for meter_threshold, deg_threshold in zip(
            [0.05, 0.15, 0.25, 0.5, 0.01, 0.02, 0.03, 0.05, 0.25, 0.5, 5],
            [2, 5, 10, 15, 1, 2, 3, 5, 2, 5, 10]
    ):
        score = torch.logical_and(
            t_errors <= meter_threshold, q_errors <= deg_threshold
        ).sum() / t_errors.shape[0]
        writer.add_scalar(
            f'{data_type} percentage localized within {meter_threshold}m, {deg_threshold}deg',
            score, epoch
        )
