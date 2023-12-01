import torch
import torch.nn.functional as F
from smplx import build_layer
from pytorch3d import transforms


def calculate_alpha(rgb_sigma, is_valid, clamp_mode, delta_alpha=0.04):
    sigmas = rgb_sigma[..., 3:]

    deltas = delta_alpha

    if clamp_mode == 'softplus':
        alphas = 1-torch.exp(-deltas * (F.softplus(sigmas)))
    elif clamp_mode == 'relu':
        alphas = 1 - torch.exp(-deltas * (F.relu(sigmas)))
    else:
        raise "Need to choose clamp mode"
    
    alphas = alphas*is_valid
    
    return alphas


def alpha_integration(rgb_alpha, z_vals, is_valid, last_back=False, white_back=False):
    rgbs = rgb_alpha[..., :3]
    alphas = rgb_alpha[..., 3:]

    alphas = alphas*is_valid
    # alphas = torch.clamp(alphas, 0, 1)

    alphas_shifted = torch.cat([torch.ones_like(alphas[:, :, :1]), 1-alphas + 1e-10], -2)
    T = torch.cumprod(alphas_shifted, -2)[:, :, :-1]
    weights = alphas * T
    weights_sum = weights.sum(2)

    if last_back:
        weights[:, :, -1] += (1 - weights_sum)

    rgb_final = torch.sum(weights * rgbs, -2)
    depth_final = torch.sum(weights * z_vals, -2)/weights_sum

    if white_back:
        rgb_final = rgb_final + 1-weights_sum

    return rgb_final, depth_final, weights, T


def convert_eulers_to_pose(joint_eulers):
    # joint_eulers: [12, 13, 14, 15, 16, 17]
    b, device = joint_eulers.shape[0], joint_eulers.device
    init_pose = torch.eye(3).reshape([1, 1, 3, 3]).repeat([b, 24, 1, 1]).to(device)
    init_pose[:, 12: 18] = transforms.euler_angles_to_matrix(joint_eulers, "ZYX")
    return init_pose


def adjust_eulers(joint_eulers, selected_inds, adjust_eulers):
    # joint_eulers: [12, 13, 14, 15, 16, 17]
    # adjust_eulers: nx3, n equals to len(selected_inds) 
    joint_eulers = joint_eulers.clone() # to avoid override original data
    b, device = joint_eulers.shape[0], joint_eulers.device
    ori_eulers = joint_eulers[:, selected_inds]
    ori_mat = transforms.euler_angles_to_matrix(ori_eulers, "ZYX")
    adjust_mat = transforms.euler_angles_to_matrix(adjust_eulers, "ZYX")
    new_mat = torch.bmm(ori_mat.reshape([-1, 3, 3]), 
                        adjust_mat.to(device).unsqueeze(0).repeat(b, 1, 1, 1).reshape([-1, 3, 3])).reshape([b, -1, 3, 3])
    new_eulers = transforms.matrix_to_euler_angles(new_mat, "ZYX")
    joint_eulers[:, selected_inds] = new_eulers
    return joint_eulers


def init_smpl(model_folder, model_type, gender, num_betas, device='cuda'):
    if device == 'cuda':
        smpl_model = build_layer(
            model_folder, model_type = model_type,
            gender = gender, num_betas = num_betas
        ).cuda()
    elif device == 'cpu':
        smpl_model = build_layer(
            model_folder, model_type = model_type,
            gender = gender, num_betas = num_betas
        )
    return smpl_model


