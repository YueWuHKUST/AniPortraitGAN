"""
The renderer is a module that takes in rays, decides where to sample along each
ray, and computes pixel colors using the volume rendering equation.
"""
import pickle 
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np 
from samplers.utils import get_initial_rays_trig_full, transform_sampled_points
from generators.utils import init_smpl, convert_eulers_to_pose, adjust_eulers
from pytorch3d.ops.knn import knn_points
from smplx.lbs import transform_mat


class Renderer(torch.nn.Module):
    def __init__(self, canonic_euler, smoother=None):
        super().__init__()
        self.num_samples = 64
        self.softmax_func = nn.Softmax(dim=-1)
        self.smpl_model = init_smpl(
            model_folder = 'smpl_models',
            model_type = 'smpl',
            gender = 'neutral',
            num_betas = 10
        )
        self.parents = self.smpl_model.parents.cpu().numpy()
        self.num_joints = self.parents.shape[0]

        self.index = 0
        
        self.valid_smpl_knn_index = np.load("./smpl_models/valid_smpl_knn_index.npy")
        self.use_symmetrical_body_space = True

        # Define the canonical space of body branch
        if canonic_euler == 't_pose':
            self.adjust_eulers_inv = torch.tensor(np.array([
                [0, 0, 0.15536945],
                [0, 0, 0],
                [0, 0, 0],
            ]), dtype=torch.float32)
        else:
            self.adjust_eulers_inv = torch.tensor(np.array([
                [0, 0, 0.15536945],
                [-1.05, 0, 0.13],
                [1.05, 0, 0.13],
            ]), dtype=torch.float32)
        
        self.selected_inds = [3, 4, 5] # 0 1 2 -> 12 13 14
        self.identity_eulers = torch.zeros((6, 3))
        self.identity_eulers[self.selected_inds, :] = self.adjust_eulers_inv
        self.identity_eulers = self.identity_eulers.unsqueeze(0)
        self.identity_pose = convert_eulers_to_pose(self.identity_eulers)
        self.smooth_layer = smoother
        self.num_samples = 64

        with open("./smpl_models/smpl/smpl_01_fit_reassign.pkl", 'rb') as fd:
            smpl_facial_barys = pickle.load(fd)['bary']
            self.smpl_facial_barys_tensor = torch.tensor(smpl_facial_barys, dtype=torch.float32)


    def sample_rays(self, batchsize, img_size, device, fov, ray_start, ray_end, camera_origin):
        '''
        functions of sampling rays. The ray start and ray end specifics the range of rays in y direction
        '''
        with torch.no_grad():
            points_cam, _, rays_d_cam = get_initial_rays_trig_full(batchsize, self.num_samples, 
                                                                        resolution=(img_size, img_size), 
                                                                        device=device, fov=fov, 
                                                                        ray_start=ray_start, ray_end=ray_end) 
            transformed_points, transformed_ray_directions, transformed_ray_origins = transform_sampled_points(
                                                                                points_cam, rays_d_cam, camera_origin, device=device)
            transformed_points = transformed_points.reshape(batchsize, -1, self.num_samples, 3)
        return transformed_points, transformed_ray_directions, transformed_ray_origins


    def forward(self, shape_z, camera_origin, given_pose, img_size, rendering_options):
        # Given pose is the new pose in the canonical space, for potential discriminator condition

        fov, ray_start, ray_end = rendering_options['fov'], rendering_options['ray_start'], rendering_options['ray_end']

        # All the points all go through the deformation network
        batchsize, device = camera_origin.shape[0], camera_origin.device
        transformed_points, ray_direction, transformed_ray_origin = self.sample_rays(batchsize, img_size, device, fov, ray_start, ray_end, camera_origin)

        origin_offset_ = torch.tensor(rendering_options['origin_offset'], dtype=torch.float32, device=device).reshape(1, 3).repeat(batchsize, 1).float()
        transformed_points = transformed_points + origin_offset_.unsqueeze(1).unsqueeze(1).repeat(1, img_size * img_size, self.num_samples, 1)
        transformed_ray_origin = transformed_ray_origin + origin_offset_.unsqueeze(1).repeat(1, img_size * img_size, 1)

        
        sample_coordinates_smpl, _ = self.run_model_lbs(shape_z, transformed_points, given_pose)


        return transformed_points, transformed_ray_origin, ray_direction, sample_coordinates_smpl


    def batch_rigid_transform(self, rot_mats, init_J):
        # joints = torch.from_numpy(init_J.reshape(1, -1, 3, 1)).cuda()
        batch_size = init_J.size()[0]
        joints = init_J.reshape(batch_size, -1, 3, 1)
        parents = self.parents

        rel_joints = joints.clone()
        rel_joints[:, 1:] -= joints[:, parents[1:]]

        transforms_mat = transform_mat(
            rot_mats.reshape(-1, 3, 3),
            rel_joints.reshape(-1, 3, 1)).reshape(-1, joints.shape[1], 4, 4)

        transform_chain = [transforms_mat[:, 0]]
        for i in range(1, parents.shape[0]):
            curr_res = torch.matmul(transform_chain[parents[i]],
                                    transforms_mat[:, i])
            transform_chain.append(curr_res)

        transforms = torch.stack(transform_chain, dim=1)

        posed_joints = transforms[:, :, :3, 3]

        joints_homogen = F.pad(joints, [0, 0, 0, 1])

        rel_transforms = transforms - F.pad(
            torch.matmul(transforms, joints_homogen), [3, 0, 0, 0, 0, 0, 0, 0])

        return posed_joints, rel_transforms


    def run_model_lbs(self, shape_z, sample_coordinates, joint_eulers):
        device = shape_z.device
        # given pose is the joint pose in new canonical space
        
        batch_size = shape_z.size()[0]
        face_ty = 0.067489409707008
        face_tz = 0.050608165033568
        offset_face = torch.from_numpy(np.array([0, face_ty, face_tz])).to(shape_z.device).float()

        face_radius = 0.52767998683149
        scale = 2.2314079
        
        joint_pose = convert_eulers_to_pose(joint_eulers)
        joint_eulers_ori = adjust_eulers(joint_eulers.clone(), self.selected_inds, self.adjust_eulers_inv)
        joint_pose_ori = convert_eulers_to_pose(joint_eulers_ori)   

        ident = self.identity_pose.repeat(batch_size, 1, 1, 1).to(device=shape_z.device)
        
        smpl_posed = self.smpl_model(betas = shape_z.reshape(batch_size, 10), body_pose = joint_pose_ori[:, 1:], global_orient = joint_pose_ori[:, :1])
        smpl_new_canonic = self.smpl_model(betas = shape_z.reshape(batch_size, 10), body_pose= ident[:, 1:].to(device), global_orient = ident[:, :1].to(device))
        
        smpl_new_canonic_j = smpl_new_canonic['joints'].clone()[:, :24]
        
        smpl_posed_verts = smpl_posed['vertices'].clone() # bs * n * 3
        smpl_posed_joints = smpl_posed['joints'].clone()[:, :24] # bs * 24 * 3
        assert smpl_posed_joints.shape[1] == 24

        # realign the smpl mesh
        del smpl_posed, smpl_new_canonic

        # recenter the smpl_posed_vertex
        smpl_posed_verts = (smpl_posed_verts - smpl_posed_joints[:, 12:13, ...].repeat(1, smpl_posed_verts.size()[1], 1)) / scale
        
        # recenter the canonical space joints
        init_J = smpl_new_canonic_j
        init_J = (init_J - init_J[:, 12:13, :]) / scale
        
        bs, n_vertex, _ = smpl_posed_verts.size()
        smpl_posed_verts = (smpl_posed_verts - offset_face.unsqueeze(0).unsqueeze(0).repeat(bs, n_vertex, 1)) / face_radius
        
        bs, n_joint, c = init_J.size()
        init_J = (init_J - offset_face.unsqueeze(0).unsqueeze(0).repeat(bs, n_joint, 1)) / face_radius
        
        _, rel_transforms = self.batch_rigid_transform(joint_pose, init_J)
        smpl_v_inv = torch.matmul(self.smpl_model.lbs_weights.reshape(-1, self.num_joints).unsqueeze(0).repeat(batch_size, 1, 1), rel_transforms.reshape(batch_size, self.num_joints, 16)).reshape(batch_size, -1, 4, 4) # 6890 * 24, 1*24*16
        smpl_v_inv = torch.inverse(smpl_v_inv.float())[:, self.valid_smpl_knn_index, :, :]
        
        N, N_rays = sample_coordinates.size()[1], sample_coordinates.size()[2]
        sample_coordinates_knn = sample_coordinates.clone()
        ori_size = int(np.sqrt(N))
        sample_coordinates_knn = sample_coordinates_knn.reshape(
            batch_size, ori_size, ori_size, N_rays, 3 
        )[:, ::2, ::2, ::2].reshape(batch_size, -1, N_rays // 2, 3)

        N_knn, N_rays_knn = sample_coordinates_knn.size()[1], sample_coordinates_knn.size()[2]
        knn_size = int(np.sqrt(N_knn))
        
        smpl_posed_verts_valid = smpl_posed_verts.reshape(batch_size, -1, 3)[:, self.valid_smpl_knn_index, :].clone()
        K = 1
        nn_pts = knn_points(sample_coordinates_knn.reshape(batch_size, -1, 3), smpl_posed_verts_valid, K=K)
        interp_weights = 1 / nn_pts.dists.reshape(batch_size, -1, K, 1, 1)
        interp_weights[torch.where(torch.isinf(interp_weights))] = 100086
        interp_weights = interp_weights / interp_weights.sum(-3, keepdim=True)

        per_point_inv_transformation = smpl_v_inv.reshape(batch_size, -1, 4, 4)
        gather_inv_T = torch.gather(per_point_inv_transformation.reshape(batch_size, -1, 1, 4, 4).repeat(1, 1, K, 1, 1), 1, nn_pts.idx.reshape(batch_size, -1, K, 1, 1).repeat(1, 1, 1, 4, 4))
        
        inv_T = (gather_inv_T * interp_weights).sum(-3).reshape(batch_size, -1, 4, 4)
        inv_T_reshape = inv_T.reshape(batch_size, knn_size, knn_size, N_rays_knn, 16).permute(0, 4, 3, 1, 2)
        inv_T_reshape_smoothed = self.smooth_layer(inv_T_reshape)
        inv_T = inv_T_reshape_smoothed.permute(0, 3, 4, 2, 1).reshape(batch_size, -1, 16).reshape(batch_size, -1, 4, 4)

        homogen_coord = torch.ones([batch_size, N*N_rays, 1], dtype=sample_coordinates.dtype, device=sample_coordinates.device)
        sample_coordinates_homo = torch.cat([sample_coordinates.view(batch_size, -1, 3), homogen_coord], dim=2)
        sample_coordinates_smpl = torch.matmul(inv_T, torch.unsqueeze(sample_coordinates_homo, dim=-1))[:, :, :3, 0].view(batch_size, -1, 3)
        
        return sample_coordinates_smpl, inv_T
