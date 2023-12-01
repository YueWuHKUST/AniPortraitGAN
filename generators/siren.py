import numpy as np
import torch.nn as nn
import torch
import math
import torch.nn.functional as F
from scipy.io import savemat
from torch.utils.checkpoint import checkpoint

class Sine(nn.Module):
    """Sine Activation Function."""
    def __init__(self):
        super().__init__()
    def forward(self, x, freq=25.):
        return torch.sin(freq * x)

def sine_init(m):
    with torch.no_grad():
        # if hasattr(m, 'weight'):
        if isinstance(m, nn.Linear):
            num_input = m.weight.size(-1)
            m.weight.uniform_(-np.sqrt(6 / num_input) / 30, np.sqrt(6 / num_input) / 30)


def first_layer_sine_init(m):
    with torch.no_grad():
        # if hasattr(m, 'weight'):
        if isinstance(m, nn.Linear):
            num_input = m.weight.size(-1)
            m.weight.uniform_(-1 / num_input, 1 / num_input)


def film_sine_init(m):
    with torch.no_grad():
        # if hasattr(m, 'weight'):
        if isinstance(m, nn.Linear):
            num_input = m.weight.size(-1)
            m.weight.uniform_(-np.sqrt(6 / num_input) / 30, np.sqrt(6 / num_input) / 30)


def first_layer_film_sine_init(m):
    with torch.no_grad():
        # if hasattr(m, 'weight'):
        if isinstance(m, nn.Linear):
            num_input = m.weight.size(-1)
            m.weight.uniform_(-1 / num_input, 1 / num_input)


def kaiming_leaky_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1 or classname.find('Conv3d') != -1:
        torch.nn.init.kaiming_normal_(m.weight, a=0.2, mode='fan_in', nonlinearity='leaky_relu')


class MappingNetwork(nn.Module):
    def __init__(self, z_dim, map_hidden_dim, map_output_dim):
        super().__init__()

        self.network = nn.Sequential(nn.Linear(z_dim, map_hidden_dim),
                                     nn.LeakyReLU(0.2, inplace=True),

                                    nn.Linear(map_hidden_dim, map_hidden_dim),
                                    nn.LeakyReLU(0.2, inplace=True),

                                    nn.Linear(map_hidden_dim, map_hidden_dim),
                                    nn.LeakyReLU(0.2, inplace=True),

                                    nn.Linear(map_hidden_dim, map_output_dim))

        self.network.apply(kaiming_leaky_init)
        with torch.no_grad():
            self.network[-1].weight *= 0.25

    def forward(self, z):
        frequencies_offsets = self.network(z)
        frequencies = frequencies_offsets[..., :frequencies_offsets.shape[-1]//2]
        phase_shifts = frequencies_offsets[..., frequencies_offsets.shape[-1]//2:]

        return frequencies, phase_shifts


class HyperMappingNetwork(nn.Module):
    def __init__(self, z_dim, map_hidden_dim, map_output_dim_freq,map_output_dim_phase,scale=0.25):
        super().__init__()

        self.map_output_dim_freq = map_output_dim_freq
        self.map_output_dim_phase = map_output_dim_phase
        self.scale = scale

        self.network = nn.Sequential(nn.Linear(z_dim, map_hidden_dim),
                                     nn.LeakyReLU(0.2, inplace=True),

                                    nn.Linear(map_hidden_dim, map_hidden_dim),
                                    nn.LeakyReLU(0.2, inplace=True),

                                    nn.Linear(map_hidden_dim, map_hidden_dim),
                                    nn.LeakyReLU(0.2, inplace=True),

                                    nn.Linear(map_hidden_dim, map_output_dim_freq+map_output_dim_phase))

        self.network.apply(kaiming_leaky_init)
        with torch.no_grad():
            self.network[-1].weight *= 0.25

    def forward(self, z):
        frequencies_offsets = self.network(z)
        # frequencies_offsets *= self.scale
        frequencies = frequencies_offsets[..., :self.map_output_dim_freq]
        phase_shifts = frequencies_offsets[..., self.map_output_dim_freq:]

        return frequencies, phase_shifts


def frequency_init(freq):
    def init(m):
        with torch.no_grad():
            # if hasattr(m, 'weight'):
            if isinstance(m, nn.Linear):
                num_input = m.weight.size(-1)
                m.weight.uniform_(-np.sqrt(6 / num_input) / freq, np.sqrt(6 / num_input) / freq)
    return init


class FiLMLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, activation='sin'):
        super().__init__()
        self.layer = nn.Linear(input_dim, hidden_dim)
        self.activation = activation

    def forward(self, x, freq, phase_shift=None,random_phase=False):
        x = self.layer(x)
        if not freq.shape == x.shape:
            freq = freq.unsqueeze(1).expand_as(x)
        if not phase_shift is None and not phase_shift.shape == x.shape:
            phase_shift = phase_shift.unsqueeze(1).expand_as(x)
        if phase_shift is None:
            phase_shift = 0
        if self.activation == 'sin':
            if random_phase:
                phase_shift = phase_shift*torch.randn(x.shape[0],x.shape[1],1).to(x.device)
            return torch.sin(freq * x + phase_shift)
        else:
            return F.leaky_relu(freq * x + phase_shift, negative_slope=0.2)


class UniformBoxWarp(nn.Module):
    def __init__(self, sidelength):
        super().__init__()
        self.scale_factor = 2/sidelength
        
    def forward(self, coordinates):
        return coordinates * self.scale_factor
    
    def forward_var(self, variances):
        return variances * self.scale_factor**2
    
    def forward_inv(self,coordinates):
        return coordinates / self.scale_factor


class RadianceNet(nn.Module):
    def __init__(self, input_dim=2, z_dim=100, hidden_dim=256, output_dim=1, feature_dim=0):
        super().__init__()
        self.input_dim = input_dim
        self.z_dim = z_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.feature_dim = feature_dim
        
        self.network = nn.ModuleList([
            FiLMLayer(3, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
        ])

        self.output_sigma = nn.ModuleList([
            nn.Linear(hidden_dim, 1),
            nn.Linear(hidden_dim, 1),
            nn.Linear(hidden_dim, 1),
            nn.Linear(hidden_dim, 1),
            nn.Linear(hidden_dim, 1),
            nn.Linear(hidden_dim, 1),
            nn.Linear(hidden_dim, 1),
        ])
        
        self.color_layer_sine = nn.ModuleList([FiLMLayer(hidden_dim + 3, hidden_dim)])

        self.output_color = nn.ModuleList([
            nn.Linear(hidden_dim, 3),
            nn.Linear(hidden_dim, 3),
            nn.Linear(hidden_dim, 3),
            nn.Linear(hidden_dim, 3),
            nn.Linear(hidden_dim, 3),
            nn.Linear(hidden_dim, 3),
            nn.Linear(hidden_dim, 3),
        ])

        if (self.feature_dim > 0): 
            self.output_feature = nn.ModuleList([
            nn.Linear(hidden_dim, self.feature_dim),
            nn.Linear(hidden_dim, self.feature_dim),
            nn.Linear(hidden_dim, self.feature_dim),
            nn.Linear(hidden_dim, self.feature_dim),
            nn.Linear(hidden_dim, self.feature_dim),
            nn.Linear(hidden_dim, self.feature_dim),
            nn.Linear(hidden_dim, self.feature_dim),
        ])
        
        # z_id and z_noise, 80 + 80
        self.mapping_network = MappingNetwork(80 * 2, 256, (len(self.network) + len(self.color_layer_sine))*hidden_dim*2)
        
        self.network.apply(frequency_init(25))
        self.output_sigma.apply(frequency_init(25))
        self.color_layer_sine.apply(frequency_init(25))
        self.output_color.apply(frequency_init(25))
        self.network[0].apply(first_layer_film_sine_init)
        
        self.gridwarper = UniformBoxWarp(0.24) # Don't worry about this, it was added to ensure compatibility with another model. Shouldn't affect performance.

    def forward(self, input, z, ray_directions, freq=None, phase=None):
        if freq is None:
            frequencies, phase_shifts = self.mapping_network(z)
            return self.forward_with_frequencies_phase_shifts(input, frequencies, phase_shifts, ray_directions)
        else:
            return self.forward_with_frequencies_phase_shifts(input, frequencies=freq, phase_shifts=phase, ray_directions=ray_directions)
    
    def forward_with_frequencies_phase_shifts(self, input, frequencies, phase_shifts, ray_directions):
        frequencies = frequencies*15 + 30
        
        input = self.gridwarper(input)
        x = input
        sigma = 0
        rgb = 0
            
        for index, layer in enumerate(self.network):
            start = index * self.hidden_dim
            end = (index+1) * self.hidden_dim
            x = layer(x, frequencies[..., start:end], phase_shifts[..., start:end])
            if index > 0:
                layer_sigma = self.output_sigma[index-1](x)
                if not index == 7:
                    layer_rgb_feature = x 
                else:
                    layer_rgb_feature = self.color_layer_sine[0](torch.cat([ray_directions, x], dim=-1),\
                        frequencies[..., len(self.network)*self.hidden_dim:(len(self.network)+1)*self.hidden_dim], phase_shifts[..., len(self.network)*self.hidden_dim:(len(self.network)+1)*self.hidden_dim])
                layer_rgb = self.output_color[index-1](layer_rgb_feature)

                sigma += layer_sigma
                rgb += layer_rgb

        rgb = torch.sigmoid(rgb)
        return torch.cat([rgb, sigma], dim=-1)

    def forward_feature_with_frequencies_phase_shifts(self, input, frequencies, phase_shifts, ray_directions):
        frequencies = frequencies*15 + 30
        
        input = self.gridwarper(input)
        x = input
        sigma = 0
        rgb = 0
        feature = 0
            
        for index, layer in enumerate(self.network):
            start = index * self.hidden_dim
            end = (index+1) * self.hidden_dim
            x = layer(x, frequencies[..., start:end], phase_shifts[..., start:end])
            if index > 0:
                if self.feature_dim > 0:
                    layer_feature = self.output_feature[index-1](x)
                    feature += layer_feature
                layer_sigma = self.output_sigma[index-1](x)
                if not index == 7:
                    layer_rgb_feature = x 
                else:
                    layer_rgb_feature = self.color_layer_sine[0](torch.cat([ray_directions, x], dim=-1),\
                        frequencies[..., len(self.network)*self.hidden_dim:(len(self.network)+1)*self.hidden_dim], phase_shifts[..., len(self.network)*self.hidden_dim:(len(self.network)+1)*self.hidden_dim])
                layer_rgb = self.output_color[index-1](layer_rgb_feature)

                sigma += layer_sigma
                rgb += layer_rgb

        rgb = torch.sigmoid(rgb)
        return torch.cat([rgb, sigma], dim=-1), feature

def constant_init_last_layer(m):
    with torch.no_grad():
        if isinstance(m, nn.Linear):
            nn.init.constant_(m.weight, 0)
            nn.init.constant_(m.bias,0)

def geometry_init(m):
    with torch.no_grad():
        # if hasattr(m, 'weight'):
        if isinstance(m, nn.Linear):
            num_output = m.weight.size(0)
            m.weight.normal_(0,np.sqrt(2/num_output))
            nn.init.constant_(m.bias,0)

def geometry_init_last_layer(radius):
    def init(m):
        with torch.no_grad():
            # if hasattr(m, 'weight'):
            if isinstance(m, nn.Linear):
                num_input = m.weight.size(-1)
                nn.init.constant_(m.weight,10*np.sqrt(np.pi/num_input))
                nn.init.constant_(m.bias,-radius)
    return init


class SampleNet(nn.Module):
    def __init__(self, input_dim=3, z_dim=100, hidden_dim=64, output_dim=1):
        super().__init__()
        self.input_dim = input_dim
        self.z_dim = z_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        self.network = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )

        self.output_layer = nn.Linear(hidden_dim, 1)

        self.center = torch.tensor([0,0,-1.5])
        self.gridwarper = UniformBoxWarp(0.24) # Don't worry about this, it was added to ensure compatibility with another model. Shouldn't affect performance.

    def calculate_intersection(self, intervals, vals, levels):
        intersections = []
        is_valid = []
        for interval,val,l in zip(intervals,vals,levels):
            x_l = interval[:,:,0]
            x_h = interval[:,:,1]
            s_l = val[:,:,0]
            s_h = val[:,:,1]
            scale = torch.where(torch.abs(s_h-s_l) > 0.05, s_h-s_l, torch.ones_like(s_h)*0.05)
            intersect = torch.where(((s_h-l<=0)*(l-s_l<=0)) & (torch.abs(s_h-s_l) > 0.05),((s_h-l)*x_l + (l-s_l)*x_h)/scale,x_h)
            # if distance is too small, choose the right point
            intersections.append(intersect)
            is_valid.append(((s_h-l<=0)*(l-s_l<=0)).to(intersect.dtype))
        return torch.stack(intersections,dim=-2),torch.stack(is_valid,dim=-2) #[batch,N_rays,level,3]

    def calculate_intersections_withlbs_with_deform(self, x_target_withexp_withhead_interval, x_target_with_exp_interval, vals, levels, deform_interval):
        intersections_target_withexp_withhead_list = []
        intersections_target_withexp_list = []
        intersections_canonic_list = []
        is_valid = []

        for interval_withexp_withhead, interval_withexp, val, l, deform in zip(x_target_withexp_withhead_interval, x_target_with_exp_interval, vals, levels, deform_interval):
            x_withexp_withhead_l = interval_withexp_withhead[:,:,0]
            x_withexp_withhead_h = interval_withexp_withhead[:,:,1]
            x_withexp_l = interval_withexp[:, :, 0]
            x_withexp_h = interval_withexp[:, :, 1]
            
            s_l = val[:,:,0]
            s_h = val[:,:,1]
            deform_l = deform[:, :, 0]
            deform_h = deform[:, :, 1]

            scale = torch.where(torch.abs(s_h-s_l) > 0.05, s_h-s_l, torch.ones_like(s_h)*0.05)
            mask = ((s_h-l<=0)*(l-s_l<=0)) & (torch.abs(s_h-s_l) > 0.05)
            intersect_target_withexp_withhead = torch.where(mask == True, ((s_h-l)*x_withexp_withhead_l + (l-s_l)*x_withexp_withhead_h)/scale, x_withexp_withhead_h)
            intersect_target_withexp =          torch.where(mask == True, ((s_h-l)*x_withexp_l          + (l-s_l)*x_withexp_h)/scale,          x_withexp_h)
            avg_deform =                        torch.where(mask == True, ((s_h-l)*deform_l             + (l-s_l)*deform_h)/scale,             deform_h)
            intersect_canonical = intersect_target_withexp + avg_deform
            
            intersections_target_withexp_withhead_list.append(intersect_target_withexp_withhead)
            intersections_target_withexp_list.append(intersect_target_withexp)
            intersections_canonic_list.append(intersect_canonical)

            is_valid.append(((s_h-l<=0)*(l-s_l<=0)).to(intersect_target_withexp_withhead.dtype))
        
        return torch.stack(intersections_target_withexp_withhead_list, dim=-2), torch.stack(intersections_target_withexp_list, dim=-2), torch.stack(intersections_canonic_list, dim=-2), torch.stack(is_valid, dim=-2)

    def forward(self,input):
        x = input
        x = self.network(x)
        s = self.output_layer(x)
        return s

    def get_s_and_intersections_target_lbs_with_deform(self, x_target_withexp_withhead, x_target_with_exp, x_canonic_manifolds, x_target_with_exp_deform_vector, levels):
        batch, N_rays, N_points, _ = x_canonic_manifolds.shape
        
        x_canonic_manifolds = x_canonic_manifolds.reshape(batch,-1,3)
        x_canonic_manifolds = self.gridwarper(x_canonic_manifolds)

        # This is because the iossurfaces of SIREN is two sides. move them to use half of the isosurfaces
        x_canonic_manifolds = x_canonic_manifolds - self.center.to(x_canonic_manifolds.device)

        # use a light wight MLP network to process a point x, and predict a scalar value s
        x_canonic_manifolds = self.network(x_canonic_manifolds)
        s = self.output_layer(x_canonic_manifolds)
        s = s.reshape(batch, N_rays, N_points, 1)
        s_l = s[:,:,:-1] 
        s_h = s[:,:,1:]

        # cost 
        cost = torch.linspace(N_points-1, 0, N_points-1).float().to(x_canonic_manifolds.device).reshape(1, 1, -1, 1)
        # shape is batch_size x 1 x [N_points - 1] x 1, ranges from N_points -1, 0

        x_target_withexp_withhead_interval = []
        x_target_with_exp_interval = []
        s_interval = []
        deform_interval = []
        for l in levels:
            r = (s_h-l <= 0) * (l-s_l <= 0) * 2 - 1
            # on the two sides of the plane, the sign is negative. when the two sides across the plane, the sign is positive.
            r = r*cost
            _, indices = torch.max(r, dim=-2, keepdim=True)
            indices_expand = indices.expand(-1, -1, -1, 3)
    
            x_target_withexp_withhead_l_select = torch.gather(x_target_withexp_withhead, -2, indices_expand) # [batch,N_rays,1]
            x_target_withexp_withhead_h_select = torch.gather(x_target_withexp_withhead, -2, indices_expand + 1) # [batch,N_rays,1]
            x_target_with_exp_l_select = torch.gather(x_target_with_exp, -2, indices_expand) # [batch,N_rays,1]
            x_target_with_exp_h_select = torch.gather(x_target_with_exp, -2, indices_expand + 1) # [batch,N_rays,1]
            deform_l_select = torch.gather(x_target_with_exp_deform_vector, -2, indices.expand(-1, -1, -1, 3)) # [batch,N_rays,1]
            deform_h_select = torch.gather(x_target_with_exp_deform_vector, -2, indices.expand(-1, -1, -1, 3) + 1) # [batch,N_rays,1]
            s_l_select = torch.gather(s_l, -2, indices)
            s_h_select = torch.gather(s_h, -2, indices)  

            x_target_withexp_withhead_interval.append(torch.cat([x_target_withexp_withhead_l_select, x_target_withexp_withhead_h_select],dim=-2))
            x_target_with_exp_interval.append(torch.cat([x_target_with_exp_l_select, x_target_with_exp_h_select],dim=-2))
            s_interval.append(torch.cat([s_l_select, s_h_select],dim=-2))
            deform_interval.append(torch.cat([deform_l_select, deform_h_select], dim=-2))
        
        intersections_target_with_exp_with_head, intersections_target_with_exp, intersections_canonical, is_valid = \
            self.calculate_intersections_withlbs_with_deform(x_target_withexp_withhead_interval, x_target_with_exp_interval, s_interval, levels, deform_interval)
        
        return intersections_target_with_exp_with_head, intersections_target_with_exp, intersections_canonical, s, is_valid


    def get_intersections(self, input, levels, **kwargs):
        # levels num_l
        batch,N_rays,N_points,_ = input.shape
        
        x = input.reshape(batch,-1,3)
        x = self.gridwarper(x)
        x = x - self.center.to(x.device)

        # use a light wight MLP network to process a point x, and predict a scalar value s
        x = self.network(x)
        s = self.output_layer(x)
        s = s.reshape(batch,N_rays,N_points,1)
        s_l = s[:,:,:-1]
        s_h = s[:,:,1:]

        # cost 
        cost = torch.linspace(N_points-1,0,N_points-1).float().to(input.device).reshape(1,1,-1,1)
        # shape is batch_size x 1 x [N_points - 1] x 1, ranges from N_points -1, 0

        x_interval = []
        s_interval = []
        for l in levels:
            r = torch.sign((s_h-l)*(l-s_l)) # [batch,N_rays,N_points-1]
            # on the two sides of the plane, the sign is negative. when the two sides across the plane, the sign is positive.
            r = r*cost
            _, indices = torch.max(r,dim=-2,keepdim=True)
            x_l_select = torch.gather(input,-2,indices.expand(-1, -1, -1, 3)) # [batch,N_rays,1]
            x_h_select = torch.gather(input,-2,indices.expand(-1, -1, -1, 3)+1) # [batch,N_rays,1]
            s_l_select = torch.gather(s_l,-2,indices)
            s_h_select = torch.gather(s_h,-2,indices)
            # gather the x coordinates and scalar
            x_interval.append(torch.cat([x_l_select,x_h_select],dim=-2))
            s_interval.append(torch.cat([s_l_select,s_h_select],dim=-2))
        
        intersections,is_valid = self.calculate_intersection(x_interval,s_interval,levels)
        
        return intersections,s,is_valid


class DeformNet(nn.Module):
    def __init__(self, input_dim=2, z_dim=100, hidden_dim=128, output_dim=1, phase_noise=False, hidden_z_dim=128):
        super().__init__()
        self.input_dim = input_dim
        self.z_dim = z_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.phase_noise = phase_noise

        self.network = nn.ModuleList([
            FiLMLayer(self.input_dim, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
        ])
        self.epoch = 0
        self.step = 0
        
        self.network.apply(frequency_init(25))
        self.network[0].apply(first_layer_film_sine_init)

        self.final_layer = nn.Linear(hidden_dim, self.output_dim)
        self.final_layer.apply(constant_init_last_layer)

        self.mapping_network = MappingNetwork(self.z_dim, hidden_z_dim, len(self.network)*hidden_dim*2)
        self.gridwarper = UniformBoxWarp(0.24) 

        self.correction_scale = 1.0
        self.deformation_scale = 1.0

    def forward(self, z_id, z_exp, coords, ref_deform, score, **kwargs):
        z = torch.cat([z_id, z_exp], dim=-1)
        batch_size, num_points, num_steps, _ = coords.shape
        batch_size = z_exp.shape[0]
        coords = coords.view(batch_size, -1, 3)
        ref_deform = ref_deform.view(batch_size, -1, 3)
        score = score.view(batch_size, -1, 1)

        coords = self.gridwarper(coords)
        ref_deform = self.gridwarper(ref_deform)
        input = torch.cat([coords, ref_deform, score], dim=-1)
        frequencies, phase_shifts = self.mapping_network(z)
        deformation = self.forward_with_frequencies_phase_shifts(input, frequencies, phase_shifts, **kwargs)
        
        new_coords = coords + deformation # deform into template space

        new_coords = new_coords.view(batch_size, num_points, num_steps, 3)
        deformation = deformation.view(batch_size, num_points, num_steps, 3)

        return new_coords, deformation

    def forward_with_frequencies_phase_shifts(self, input, frequencies, phase_shifts, **kwargs):
        frequencies = frequencies*15 + 30
        x = input
    
        for index, layer in enumerate(self.network):
            start = index * self.hidden_dim
            end = (index+1) * self.hidden_dim
            x = layer(x, frequencies[..., start:end], phase_shifts[..., start:end],random_phase=self.phase_noise)
        
        x = self.final_layer(x)
        deformation = x[...,:3]
        return deformation
    