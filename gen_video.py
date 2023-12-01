"""Generate videos given pose key frams"""

import os
import sys
sys.path.append('./FaceRecon')
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from tqdm import tqdm
import imageio
from easydict import EasyDict as edict
import cv2

import curriculums
import generators
from samplers.joint_sampler import IDShapeSampler, CameraJointSampler, ExpressionSampler
from samplers.utils import transform_from_euler_to_orgin, z_sampler

device = torch.device('cuda')


def convert_depth(depth_image):
    near = 1.7550879793728869
    far = 2.115087979372887
    depth_image = 1.0 - (depth_image - near) / (far - near)
    depth_image = (depth_image.permute(0, 2, 3, 1) * 255).clamp(0, 255).to(torch.uint8)
    return depth_image


def parse_seeds(seeds):
    seeds_str = seeds.strip().split(',')
    seeds_str = filter(lambda x: x != '', seeds_str)
    seeds = []
    for s in seeds_str:
        if '-' in s:
            seeds += list(np.arange(int(s.split('-')[0]), int(s.split('-')[1])))
        else:
            seeds.append(int(s))
    return seeds


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, default='./ckpts/sshq512.pt')
    parser.add_argument('--output_dir', type=str, default='./video')
    parser.add_argument('--curriculum', type=str, default='SSHQ512_inf')

    # body render options
    parser.add_argument('--face_radius', help='the near plane of sample space of face region', type=float, default=0.52767998683149)
    parser.add_argument('--face_ty', help='the offset along y axis of face camera', type=float, default=0.067489409707008)
    parser.add_argument('--face_tz', help='the offset along z axis of face camera', type=float, default=0.050608165033568)
    parser.add_argument('--body_ray_start', help='the offset along y axis of face camera', type=float, default=1.675087979372887)
    parser.add_argument('--body_ray_end', help='the offset along y axis of face camera', type=float, default=2.115087979372887)

    # multi options
    parser.add_argument('--seeds', type=str, default='42,', help='random seeds (either range str or npy path)')
    parser.add_argument('--seeds_interval', type=int, default=100, help='interval between seeds')
    parser.add_argument('--workers', type=int, default=1, help='total number of workers')
    parser.add_argument('--rank', type=int, default=0, help='rank')

    # video gen options
    parser.add_argument('--psi', type=float, default=0.7, help='truncation psi')
    parser.add_argument('--exp_path', type=str, default='null', help='path to expression sequence') # [N, 64]
    parser.add_argument('--pose_path', type=str, default='null', help='path to pose sequence') # [N, 6, 3]
    parser.add_argument('--cam_path', type=str, default='null', help='path to camera sequence') # [N, 2]
    parser.add_argument('--n_interval', type=int, default=30, help='Number of interpolation frames between two keyframes')
    parser.add_argument('--cam_yaw_range', type=float, default=0.0, help='camera yaw range')
    parser.add_argument('--cam_cycle', type=int, default=2, help='camera movement cycle (s)')
    parser.add_argument('--video_length', type=int, default=0, help='video length (s), 0 for auto')
    parser.add_argument('--fps', type=float, default=30, help='fps')
    parser.add_argument('--ext', type=str, default='mp4', help='video extension')
    parser.add_argument('--random_instance', action='store_true', help='random instance')
    parser.add_argument('--random_pose', type=str, default='null', help='random pose')

    opt = parser.parse_args()


    # Body render options
    origin_offset_body = np.array([0, 0.052552, 0])
    offset_face = np.array([0, opt.face_ty, opt.face_tz])
    origin_offset_body = origin_offset_body - offset_face
    origin_offset_body = origin_offset_body / opt.face_radius
    body_raidus = 1.0 / opt.face_radius
    origin_offset_body[1] += 0.01
    rendering_options = {
        'ray_start': opt.body_ray_start, # near point along each ray to start taking samples.
        'ray_end': opt.body_ray_end, # far point along each ray to stop taking samples.
        'radius': body_raidus, # radius of the sphere to render.
        'fov': 12,
        'origin_offset': origin_offset_body,
    }

    # Setup models    
    curriculum = getattr(curriculums, opt.curriculum)
    generator = generators.ManifoldSRGenerator3d(**curriculum['generator']['kwargs'])
    print("Generator ckpt:", opt.ckpt)
    generator.load_state_dict(torch.load(opt.ckpt, map_location=device), strict=True)
    generator = generator.to(device)
    generator.eval()

    joint_sampler = CameraJointSampler(edict(camera_pose_joint_sample=np.load('./sampler_npy/camera_pose_joint_sample.npy')))
    shape_sampler = IDShapeSampler(
        edict(id_sample=np.load('./sampler_npy/id_face_sampler.npy')),
        edict(id_shape_sample=np.load('./sampler_npy/id_shape_sampler_body.npy'))
    )
    exp_sampler = ExpressionSampler(
        edict(expression_sample=np.load('./sampler_npy/expression_face.npy')),
        edict(expression_sample=np.load('./sampler_npy/expression_body.npy'))
    )

    # Setup the frames
    if opt.video_length == 0:
        keyframe_file = []
        if opt.exp_path != 'null': keyframe_file.append(opt.exp_path)
        if opt.pose_path != 'null': keyframe_file.append(opt.pose_path)
        if opt.cam_path != 'null': keyframe_file.append(opt.cam_path)
        assert len(keyframe_file) > 0, \
            "video_length must be set if all of exp_path, pose_path and cam_path are null"
        keyframes = np.load(keyframe_file[0]).shape[0]
        n_interval = opt.n_interval
        frames = keyframes * n_interval
        opt.video_length = frames / opt.fps
        cam_cycles = opt.video_length // opt.cam_cycle
    else:
        assert opt.video_length % opt.cam_cycle == 0, "video_length must be divisible by cam_cycle"
        frames = int(opt.fps * opt.video_length)
        keyframes = frames // opt.n_interval
        frames_per_cam_cycle = int(opt.fps * opt.cam_cycle)
        cam_cycles = opt.video_length // opt.cam_cycle
        n_interval = opt.n_interval
    ## camera
    h_mean = np.pi * 0.5
    v_mean = np.pi * 0.5
    if opt.cam_path == 'null':
        if opt.cam_yaw_range != 0:
            yaws = list(np.linspace(-opt.cam_yaw_range, opt.cam_yaw_range, frames_per_cam_cycle // 2 + 1)[:-1]) \
                + list(np.linspace(opt.cam_yaw_range, -opt.cam_yaw_range, frames_per_cam_cycle // 2 + 1)[:-1])
            yaws = yaws * cam_cycles
        else:
            yaws = [0] * frames
        pitches = [0] * frames
        camera_angles = [[a + h_mean, b + v_mean] for a, b in zip(yaws, pitches)]
    elif opt.cam_path == 'rand':
        camera_angles = [[np.random.uniform(-np.pi, np.pi), np.random.uniform(-np.pi, np.pi)] for _ in range(frames)]
    else:
        camera_angles_np = np.load(opt.cam_path)
        camera_angles_np = camera_angles_np[:frames]
        camera_angles_np = np.concatenate([camera_angles_np, camera_angles_np[:1]], axis=0)
        camera_angles = []
        for i in range(keyframes):
            for j in range(n_interval):
                k = j / n_interval
                yaw = (1 - k) * camera_angles_np[i][1] + k * camera_angles_np[i + 1][1]
                pitch = (1 - k) * camera_angles_np[i][0] + k * camera_angles_np[i + 1][0]
                camera_angles.append([yaw, pitch])
    ## expression
    if opt.exp_path == 'null' or opt.exp_path == 'rand':
        z_exps_np = np.zeros((keyframes, 64))
    else:
        z_exps_np = np.load(opt.exp_path)
    z_exps_np = z_exps_np[:keyframes]
    z_exps_np = np.concatenate([z_exps_np, z_exps_np[:1]], axis=0)
    z_exps = []
    for i in range(keyframes):
        for j in range(n_interval):
            k = j / n_interval
            z_exps.append(torch.tensor(
                (1 - k) * z_exps_np[i] + k * z_exps_np[i + 1],
            ).float().to(device).unsqueeze(0))
    ## pose
    if opt.pose_path == 'null' or opt.pose_path == 'rand':
        z_poses_np = np.zeros((keyframes, 6, 3))
    else:
        z_poses_np = np.load(opt.pose_path)
    z_poses_np = z_poses_np[:keyframes]
    z_poses_np = np.concatenate([z_poses_np, z_poses_np[:1]], axis=0)
    z_poses = []
    for i in range(keyframes):
        for j in range(n_interval):
            k = j / n_interval
            z_poses.append(torch.tensor(
                (1 - k) * z_poses_np[i] + k * z_poses_np[i + 1],
            ).float().to(device).unsqueeze(0))

    # final misc
    output_dir = opt.output_dir
    os.makedirs(output_dir, exist_ok=True)
    psi = opt.psi
    if opt.seeds.endswith('.npy'):
        seeds = list(np.load(opt.seeds))
    elif ',' in opt.seeds or '-' in opt.seeds:
        seeds = parse_seeds(opt.seeds)
    else:
        seeds = list(np.arange(int(opt.seeds), int(opt.seeds) + opt.seeds_interval))
    if opt.workers > 1:
        seeds = seeds[len(seeds) * opt.rank // opt.workers: len(seeds) * (opt.rank + 1) // opt.workers]

    generator.generate_avg_frequencies(shape_sampler, device)

    # Video generation
    print("Generating video...")
    with torch.no_grad():
        with tqdm(total=len(seeds), desc='Total Progress', position=0, leave=True) as pbar_seeds:
            for seed in seeds:
                torch.manual_seed(seed)
                z_id, z_shape = shape_sampler.forward_body(1, device)
                z_noise = z_sampler((1, 80), device=device, dist='gaussian')
                z = torch.cat([z_id, z_noise], dim=1)

                if opt.random_instance:
                    if opt.pose_path == 'rand' or opt.cam_path == 'rand':
                        camera_pose, z_pose_ = joint_sampler.forward(1, device)
                    if opt.cam_path == 'rand':
                        camera_angles = [[camera_pose[0, 1], camera_pose[0, 0]]] * frames
                    if opt.pose_path == 'rand':
                        z_pose = z_pose_ * 0.5
                        z_poses = [z_pose] * frames
                    if opt.random_pose == 'head':
                        for z_pose in z_poses:
                            z_pose[:, [0, 3]] = z_pose_[:, [0, 3]]
                    if opt.random_pose == 'shoulder':
                        for z_pose in z_poses:
                            z_pose[:, [1, 2, 4, 5]] = z_pose_[:, [1, 2, 4, 5]]
                    if opt.exp_path == 'rand':
                        z_exp = exp_sampler.forward_body(1, device)
                        z_exps = [z_exp] * frames

                # sr and cache manifold
                raw_freq, raw_phase = generator.radiance_net.mapping_network(z)
                trunc_freq = generator.avg_frequencies * (1 - psi) + raw_freq * psi
                trunc_phase = generator.avg_phase_shifts * (1 - psi) + raw_phase * psi
                sr_output = generator.manifold_sr([z_id, None, z_noise], trunc_freq, trunc_phase, truncation_psi=psi)

                imgs_rgb = []
                imgs_rgb_lr = []
                imgs_depth = []
                
                with tqdm(total=frames, desc=f'Seed {seed}', position=1, leave=False) as pbar_frames:
                    for (yaw, pitch), z_exp, z_pose in zip(camera_angles, z_exps, z_poses):
                        camera_pose = torch.tensor([[pitch, yaw]]).to(device)
                        if not opt.random_instance:
                            if opt.pose_path == 'rand' or opt.cam_path == 'rand':
                                camera_pose_, z_pose_ = joint_sampler.forward(1, device)
                            if opt.cam_path == 'rand':
                                camera_pose = camera_pose_
                            if opt.pose_path == 'rand':
                                z_pose = z_pose_ * 0.7
                            if opt.exp_path == 'rand':
                                z_exp = exp_sampler.forward_body(1, device)
                            
                        camera_origin = transform_from_euler_to_orgin(device, camera_pose[:, :1], camera_pose[:, 1:], 1, r=rendering_options['radius'])

                        res = generator.synthesis([z_id, z_exp, z_noise], z_shape, camera_origin, z_pose, \
                                                  sr_output=sr_output, rendering_options=rendering_options)
                    
                        image = res['gen_img']
                        image = (image.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
                        image_lr = res['gen_img_lr']
                        image_lr = (image_lr.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
                        depth = convert_depth(res['depth_image'])
                        imgs_rgb.append(image[0, ...].cpu().numpy())
                        imgs_rgb_lr.append(image_lr[0, ...].cpu().numpy())
                        imgs_depth.append(cv2.cvtColor(cv2.applyColorMap(depth[0, ...,0].cpu().numpy(), cv2.COLORMAP_INFERNO), cv2.COLOR_BGR2RGB))

                        pbar_frames.update(1)

                if opt.ext == 'gif':
                    imageio.mimsave(os.path.join(output_dir, f'video_{seed}_rgb.gif'), imgs_rgb, fps=opt.fps)
                    imageio.mimsave(os.path.join(output_dir, f'video_{seed}_rgb_lr.gif'), imgs_rgb_lr, fps=opt.fps)
                    imageio.mimsave(os.path.join(output_dir, f'video_{seed}_depth.gif'), imgs_depth, fps=opt.fps)
                elif opt.ext == 'mp4':
                    imageio.mimwrite(os.path.join(output_dir, f'video_{seed}_rgb.mp4'), imgs_rgb, fps=opt.fps, quality=8)
                    imageio.mimwrite(os.path.join(output_dir, f'video_{seed}_rgb_lr.mp4'), imgs_rgb_lr, fps=opt.fps, quality=8)
                    imageio.mimwrite(os.path.join(output_dir, f'video_{seed}_depth.mp4'), imgs_depth, fps=opt.fps, quality=8)
                elif opt.ext == 'png':
                    for i in range(len(imgs_rgb)):
                        imageio.imwrite(os.path.join(output_dir, f'video_{seed}_rgb_{i:03d}.png'), imgs_rgb[i])
                        imageio.imwrite(os.path.join(output_dir, f'video_{seed}_rgb_lr_{i:03d}.png'), imgs_rgb_lr[i])
                        imageio.imwrite(os.path.join(output_dir, f'video_{seed}_depth_{i:03d}.png'), imgs_depth[i])
                
                pbar_seeds.update(1)

    