"""Manifold super-resolution generator"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from generators.siren import RadianceNet, SampleNet, DeformNet
from generators.renderer import Renderer
from generators.styleesrgan import StyleRRDBNet
from generators.utils import calculate_alpha, alpha_integration

SAMPLE_CENTER, SAMPLE_SCALE = -0.02830771, 0.22


class Conv3dBNReLu_AvgPool(nn.Module):
    # use conv to implement average pooling
    def __init__(self, k_d, in_channels=16, out_channels=16, use_bn_relu=True):
        super().__init__()
        k_sp = 2 * k_d - 1
        weight = 1 / (k_d * k_sp * k_sp) * torch.ones(in_channels, 1, k_d, k_sp, k_sp)
        self.weight = nn.parameter.Parameter(weight)
        self.out_channels = out_channels
        self.stride = (1, 1, 1)
        self.padding = (k_sp // 2, k_sp // 2, k_sp // 2, k_sp // 2, k_d // 2, k_d // 2)

        self.use_bn_relu = use_bn_relu
        if use_bn_relu:
            self.bn = nn.BatchNorm3d(out_channels)
            self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        weight_sum = self.weight.sum(dim=(2, 3, 4), keepdim=True)
        weight = self.weight / weight_sum
        x = F.pad(x, self.padding, mode='reflect')
        x = F.conv3d(x, weight, stride=self.stride, groups=self.out_channels)
        if self.use_bn_relu:
            x = self.bn(x)
            x = self.activation(x)
        return x


class LbsSmoother(nn.Module):
    def __init__(self, k_d=3, in_channels=16, out_channels=16, first_bn_relu=False):
        super().__init__()
        self.conv_block1 = Conv3dBNReLu_AvgPool(k_d, in_channels=in_channels, out_channels=out_channels, use_bn_relu=first_bn_relu)
        self.up_sample = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        self.conv_block2 = Conv3dBNReLu_AvgPool(k_d, in_channels=out_channels, out_channels=out_channels, use_bn_relu=False)
    
    def forward(self, x):
        x = self.conv_block1(x)
        x = self.up_sample(x)
        x = self.conv_block2(x)
        return x


class ManifoldSRGenerator3d(nn.Module):
    def __init__(
            self,
            z_dim,
            feature_dim,
            hidden_dim,
            hidden_dim_sample,
            hr_img_size,
            sr_net_kwargs,
            lr_img_size=None,
            **kwargs
        ):
        super().__init__()
        """
        Args:
            z_dim: dimension of the latent code
            feature_dim: dimension of the feature used for super resolution
            hidden_dim: dimension of the hidden layer in the radiance network
            hidden_dim_sample: dimension of the hidden layer in the sample network
            hr_img_size: size of the high resolution image
            lr_img_size: size of the low resolution image
            sr_net_kwargs: kwargs for the super resolution network
        """
        self.z_dim = z_dim
        self.feature_dim = feature_dim
        self.scale_factor = hr_img_size // lr_img_size if lr_img_size is not None else 4
        self.hr_img_size = hr_img_size
        self.cfg = kwargs

        self.radiance_net = RadianceNet(input_dim=3, z_dim=self.z_dim, output_dim=4, feature_dim=feature_dim, hidden_dim=hidden_dim)
        self.sample_net = SampleNet(input_dim=3, z_dim=self.z_dim, hidden_dim=hidden_dim_sample)
        self.deform_net = DeformNet(input_dim=7, z_dim=64+80, output_dim=7)
        self.sr_net = StyleRRDBNet(4 + feature_dim, 4, scale_factor = self.scale_factor, use_mapping_network=True, **sr_net_kwargs)
        self.smoother = LbsSmoother(k_d=5, in_channels=16, out_channels=16)
        self.renderer = Renderer('mean_pose', self.smoother)

    def anifacegan(self, z, x_target_space):
        """
        Deform the target space to the canonical space
        """
        z_id, z_exp, _ = z
        x_deform_ref = torch.zeros(*x_target_space.shape[:3], 3).to(z_id.device)
        x_confidence = torch.zeros(*x_target_space.shape[:3], 1).to(z_id.device)
        x_canonic_manifolds, x_exp_deform_vector = self.deform_net(z_id, z_exp, x_target_space, x_deform_ref, x_confidence)
        return x_canonic_manifolds, x_exp_deform_vector
    
    def manifold_sr(self, z, freq, phase, truncation_psi=1):
        """
        Sample and super resolution the manifolds
        """
        z_id, _, noise = z 
        B = z_id.shape[0]
        H = W = self.hr_img_size // self.scale_factor
        M = self.cfg['num_manifolds']
        device = z_id.device
        levels_start = self.cfg['levels_start']
        levels_end = self.cfg['levels_end']
        num_samples = self.cfg['num_samples']
        gridwarper = self.deform_net.module.gridwarper if hasattr(self.deform_net, 'module') else self.deform_net.gridwarper

        # Orthogonal sampling
        y, x = torch.meshgrid(torch.linspace(-SAMPLE_SCALE + SAMPLE_CENTER, SAMPLE_SCALE + SAMPLE_CENTER, H), torch.linspace(-SAMPLE_SCALE, SAMPLE_SCALE, W), indexing='ij')
        sample_ray_origins = torch.stack([x.reshape(-1), y.reshape(-1), torch.ones(H*W)], dim=-1).expand(B, -1, -1).to(device)
        sample_ray_directions = torch.tensor([0, 0, -1], device=device).expand(B, H*W, -1)
        sample_points = sample_ray_origins.unsqueeze(-2) + torch.linspace(0.8, 1.2, num_samples, device=device).unsqueeze(-1) * sample_ray_directions.unsqueeze(-2)
        sample_points = sample_points.reshape(B, H*W, -1, 3)
        levels = torch.linspace(levels_start, levels_end, M).to(device)
        
        # Compute the intersections
        sample_points = gridwarper(sample_points)  # IMPORTANT
        intersections,_,is_valid = self.sample_net.get_intersections(sample_points, levels)
        intersections = gridwarper.forward_inv(intersections)  # IMPORTANT
        intersections = intersections.reshape(B, H*W*M, 3)
        sample_ray_directions = sample_ray_directions.unsqueeze(-2).expand(-1, -1, M, -1).reshape(B, H*W*M, 3)

        # Compute the radiance field
        coarse_output, coarse_feature = self.radiance_net.forward_feature_with_frequencies_phase_shifts(intersections, freq, phase, sample_ray_directions)
        coarse_output = coarse_output.reshape(B, H*W, M, 4)
        coarse_feature = coarse_feature.reshape(B, H*W, M, -1)
        coarse_output[..., 3:] = calculate_alpha(coarse_output, is_valid, clamp_mode=self.cfg['clamp_mode'])
        coarse_feature = torch.cat([coarse_output, coarse_feature], dim=-1)

        # Super resolution
        lr_features = coarse_feature.permute(0, 2, 3, 1).reshape(B*M, -1, H, W)
        lr_imgs = coarse_output.permute(0, 2, 3, 1).reshape(B*M, 4, H, W)
        hr_imgs = self.sr_net(lr_features, torch.cat([z_id, noise], dim=1), truncation_psi=truncation_psi)
        hr_imgs = torch.sigmoid(hr_imgs)
        
        return {
            'lr_imgs': lr_imgs,
            'hr_imgs': hr_imgs,
        }

    def render(self, z_id, intersections_lr, is_valid_lr, camera_origin, sr_output):
        """
        Render the final image
        """
        B = z_id.shape[0]
        HS = WS = self.hr_img_size
        HL = WL = self.hr_img_size // self.scale_factor
        M = self.cfg['num_manifolds']
        camera_origin = camera_origin[:, None, None]
            
        # Interpolate the intersections
        intersections_lr = intersections_lr[:, :, :M]
        is_valid_lr = is_valid_lr[:, :, :M]
        intersections = F.interpolate(intersections_lr.permute(0, 2, 3, 1).reshape(B*M, 3, HL, WL), scale_factor=self.scale_factor, mode='bilinear', align_corners=True) \
            .reshape(B, M, 3, HS*WS) \
            .permute(0, 3, 1, 2) \
            .reshape(B, HS*WS, M, 3)

        is_valid = (F.interpolate(is_valid_lr.permute(0, 2, 3, 1).reshape(B*M, 1, HL, WL), scale_factor=self.scale_factor, mode='bilinear', align_corners=True) > 0.99) \
            .reshape(B, M, 1, HS*WS) \
            .permute(0, 3, 1, 2) \
            .reshape(B, HS*WS, M, 1).to(intersections.dtype)
                
        # Compute the z values
        z_vals = torch.sqrt(torch.sum((intersections - camera_origin)**2,dim=-1,keepdim=True))
        z_vals[is_valid==0] = 10.
        z_vals_lr = torch.sqrt(torch.sum((intersections_lr - camera_origin)**2,dim=-1,keepdim=True))
        z_vals_lr[is_valid_lr==0] = 10.

        # Render the coarse image
        sample_coord_lr = intersections_lr[..., :2].float().permute(0, 2, 1, 3).reshape(B*M, HL, WL, 2)
        sample_coord_lr[..., 1] -= SAMPLE_CENTER
        sample_coord_lr /= SAMPLE_SCALE
        coarse_output_lr = F.grid_sample(sr_output['lr_imgs'].float(), sample_coord_lr, mode='bilinear', padding_mode='border', align_corners=False).reshape(B, M, 4, HL*WL).permute(0, 3, 1, 2)
    
        _, indices_lr = torch.sort(z_vals_lr, dim=-2)
        z_vals_lr = torch.gather(z_vals_lr, -2, indices_lr)
        coarse_output_lr = torch.gather(coarse_output_lr, -2, indices_lr.expand(B, HL*WL, M, 4))
        is_valid_lr = torch.gather(is_valid_lr, -2, indices_lr)
        pixels_lr, _, _, _ = alpha_integration(coarse_output_lr, z_vals_lr, is_valid=is_valid_lr, \
                                    white_back=self.cfg.get('white_back', False), last_back=self.cfg.get('last_back', False))

        # Render the final image
        sample_coord = intersections[..., :2].float().permute(0, 2, 1, 3).reshape(B*M, HS, WS, 2)
        sample_coord[..., 1] -= SAMPLE_CENTER
        sample_coord /= SAMPLE_SCALE
        coarse_output = F.grid_sample(sr_output['hr_imgs'].float(), sample_coord, mode='bilinear', padding_mode='border', align_corners=False).reshape(B, M, 4, HS*WS).permute(0, 3, 1, 2)

        _, indices = torch.sort(z_vals, dim=-2)
        z_vals = torch.gather(z_vals, -2, indices)
        coarse_output = torch.gather(coarse_output, -2, indices.expand(-1, -1, -1, 4))
        is_valid = torch.gather(is_valid, -2, indices)
        pixels, depth, weights, T = alpha_integration(coarse_output, z_vals, is_valid=is_valid, \
                                        white_back=self.cfg.get('white_back', False), last_back=self.cfg.get('last_back', False))
            
        pixels = pixels.reshape((B, HS, WS, 3))
        pixels = pixels.permute(0, 3, 1, 2).contiguous() * 2 - 1
        pixels_lr = pixels_lr.reshape(B, HL, WL, 3)
        pixels_lr = pixels_lr.permute(0, 3, 1, 2).contiguous() * 2. - 1.
        depth = depth.reshape((B, HS, WS, 1))
        depth = depth.permute((0, 3, 1, 2))
        
        return {
            'pixels': pixels,
            'depth': depth,
            'weights': weights,
            'T': T,
            'pixels_lr': pixels_lr,
        }
    
    def synthesis(self, z, shape_z, c_origin, given_pose, sr_output, rendering_options):
        """
        Generate a body image

        Args:
            z: input z contains[id, exp, noise]
            c_origin: camera origin of sample body image
            given_pose: the joint pose
        """
        camera_origin = c_origin
        H = W = self.hr_img_size // self.scale_factor
        B = shape_z.size()[0]
        device = shape_z.device
        levels_start = self.cfg['levels_start']
        levels_end = self.cfg['levels_end']
        num_manifolds = self.cfg['num_manifolds']
        z_id, _, _ = z 
        
        # Sample points and compute intersections with the background
        x_target_withexp_withhead, _, _, x_target_with_exp \
            = self.renderer(shape_z, camera_origin, given_pose, H, rendering_options)

        # Interpolate the sample points
        x_target_withexp_withhead = x_target_withexp_withhead.reshape(B*H*W, -1, 3).permute(0, 2, 1)
        x_target_with_exp = x_target_with_exp.reshape(B*H*W, -1, 3).permute(0, 2, 1)
        x_target_withexp_withhead = F.interpolate(x_target_withexp_withhead, size=256, mode='linear', align_corners=True)
        x_target_with_exp = F.interpolate(x_target_with_exp, size=256, mode='linear', align_corners=True)
        x_target_withexp_withhead = x_target_withexp_withhead.permute(0, 2, 1).reshape(B, H*W, -1, 3).contiguous()
        x_target_with_exp = x_target_with_exp.permute(0, 2, 1).reshape(B, H*W, -1, 3).contiguous()
            
        # Transform x target with exp -> x canonic woexp
        levels = torch.linspace(levels_start, levels_end, num_manifolds).to(device)
        bs, N_rays, N_points, channels = x_target_withexp_withhead.size()
        x_target_with_exp = x_target_with_exp.view(bs, N_rays, N_points, channels)
        x_canonic_manifolds, x_target_with_exp_deform_vector = self.anifacegan(z, x_target_with_exp)
        _, _, intersections_canonical, _, is_valid = \
            self.sample_net.get_s_and_intersections_target_lbs_with_deform(x_target_withexp_withhead, x_target_with_exp, x_canonic_manifolds, x_target_with_exp_deform_vector, levels)

        res = self.render(z_id, intersections_canonical, is_valid, camera_origin, sr_output=sr_output)

        return_dict = {
            "gen_img": res['pixels'],
            "depth_image": res['depth'],
            "gen_img_lr": res['pixels_lr'],
        }
        return return_dict

    def generate_avg_frequencies(self, shape_sampler, device):
        with torch.no_grad():
            z_id, shape_z = shape_sampler.forward_body(10000, device)
            z_noise = torch.randn((10000, 80), device=device)
        z = torch.cat([z_id, z_noise], dim=1)
        with torch.no_grad():
            frequencies, phase_shifts = self.radiance_net.mapping_network(z)
        self.avg_frequencies = frequencies.mean(0, keepdim=True)
        self.avg_phase_shifts = phase_shifts.mean(0, keepdim=True)
        self.sr_net.get_avg_w(z)
