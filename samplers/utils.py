import torch
import numpy as np
import math

def transform_from_euler_to_orgin(device, phi, theta, n=1, r=1):
    # phi - pitch, theta - yaw
    camera_origin = torch.zeros((n, 3), device=device)# torch.cuda.FloatTensor(n, 3).fill_(0)#torch.zeros((n, 3))

    camera_origin[:, 0:1] = r*torch.sin(phi) * torch.cos(theta)
    camera_origin[:, 2:3] = r*torch.sin(phi) * torch.sin(theta)
    camera_origin[:, 1:2] = r*torch.cos(phi)
    return camera_origin


def normalize_vecs(vectors: torch.Tensor) -> torch.Tensor:
    """
    Normalize vector lengths.
    """
    return vectors / (torch.norm(vectors, dim=-1, keepdim=True))


def perturb_points(points, z_vals, ray_directions, device):
    distance_between_points = z_vals[:,:,1:2,:] - z_vals[:,:,0:1,:]
    offset = (torch.rand(z_vals.shape, device=device)-0.5) * distance_between_points
    z_vals = z_vals + offset

    points = points + offset * ray_directions.unsqueeze(2)
    return points, z_vals


def get_initial_rays_trig_full(n, num_samples, device, fov, resolution, ray_start, ray_end, randomize=False):
    """Returns sample points, z_vals, ray directions in the full image space."""
    W, H = resolution
    # Create full screen NDC (-1 to +1) coords [x, y, 0, 1].
    # Y is flipped to follow image memory layouts.
    x, y = torch.meshgrid(torch.linspace(-1, 1, W, device=device),
                          torch.linspace(1, -1, H, device=device), indexing='ij')
    
    x = x.T.flatten()
    y = y.T.flatten()

    # use the camera as the coordinate origin, the rays from the camera to the world is the negative direction
    z = -torch.ones_like(x, device=device) / np.tan((2 * math.pi * fov / 360)/2)

    rays_d_cam = normalize_vecs(torch.stack([x, y, z], -1))
    z_vals = torch.linspace(ray_start, ray_end, num_samples, device=device).reshape(1, num_samples, 1).repeat(rays_d_cam.shape[0], 1, 1)
    points = rays_d_cam.unsqueeze(1).repeat(1, num_samples, 1) * z_vals

    points = torch.stack(n*[points])
    z_vals = torch.stack(n*[z_vals])
    rays_d_cam = torch.stack(n*[rays_d_cam]).to(device)
    
    if randomize:
        perturb_points(points, z_vals, rays_d_cam, device)
    return points, z_vals, rays_d_cam


def transform_sampled_points(points, ray_directions, camera_origin, device):
    n, num_rays, num_samples, channels = points.shape
    forward_vector = normalize_vecs(-camera_origin)
    #print("forward_vector", forward_vector, camera_origin, camera_origin)
    cam2world_matrix = create_cam2world_matrix(forward_vector, camera_origin, device=device)

    points_homogeneous = torch.ones((points.shape[0], points.shape[1], points.shape[2], points.shape[3] + 1), device=device)
    points_homogeneous[:, :, :, :3] = points

    # should be n x 4 x 4 , n x r^2 x num_samples x 4
    transformed_points = torch.bmm(cam2world_matrix, points_homogeneous.reshape(n, -1, 4).permute(0,2,1)).permute(0, 2, 1).reshape(n, num_rays, num_samples, 4)
    #print("cam2world", cam2world_matrix)

    transformed_ray_directions = torch.bmm(cam2world_matrix[..., :3, :3], ray_directions.reshape(n, -1, 3).permute(0,2,1)).permute(0, 2, 1).reshape(n, num_rays, 3)

    homogeneous_origins = torch.zeros((n, 4, num_rays), device=device)
    homogeneous_origins[:, 3, :] = 1
    transformed_ray_origins = torch.bmm(cam2world_matrix, homogeneous_origins).permute(0, 2, 1).reshape(n, num_rays, 4)[..., :3]
    return transformed_points[..., :3], transformed_ray_directions, transformed_ray_origins


def create_cam2world_matrix(forward_vector, origin, device=None):
    """Takes in the direction the camera is pointing and the camera origin and returns a world2cam matrix."""
    forward_vector = normalize_vecs(forward_vector)
    up_vector = torch.tensor([0, 1, 0], dtype=torch.float, device=device).expand_as(forward_vector)

    #print("up_vector, forward_vector", up_vector, forward_vector)
    left_vector = normalize_vecs(torch.cross(up_vector, forward_vector, dim=-1))

    up_vector = normalize_vecs(torch.cross(forward_vector, left_vector, dim=-1))

    #print("debug left", left_vector, up_vector, forward_vector)
    rotation_matrix = torch.eye(4, device=device).unsqueeze(0).repeat(forward_vector.shape[0], 1, 1)
    rotation_matrix[:, :3, :3] = torch.stack((-left_vector, up_vector, -forward_vector), axis=-1)

    translation_matrix = torch.eye(4, device=device).unsqueeze(0).repeat(forward_vector.shape[0], 1, 1)
    translation_matrix[:, :3, 3] = origin

    cam2world = translation_matrix @ rotation_matrix
    #print("debug cam2world inside", cam2world, translation_matrix, rotation_matrix)
    return cam2world


def z_sampler(shape, device, dist):
    if dist == 'gaussian':
        z = torch.randn(shape, device=device)
        # torch.randn - sample random numbers from a normal distribution with mean 0 and varaiance 1
    elif dist == 'uniform':
        z = torch.rand(shape, device=device) * 2 - 1
        # torch.rand - sample random numbers froma uniform distribution 
    return z