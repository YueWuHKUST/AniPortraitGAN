"""
The joint sampler is a sampler function for joint rotation
"""
import torch


class JointSampler(torch.nn.Module):
    """
    The sampling of joint euler rotations
    """
    def __init__(self, sample_method, training_dataset):
        super().__init__()
        self.sample_method = sample_method
        self.training_dataset = training_dataset
        self.euler_sample = torch.from_numpy(training_dataset.euler_sample)

    def forward(self, n, device, better_vis=False):
        """
        # the mean and std of joint 12 and joint 15
        # zyx order
        # return the rotation matrix of pt12 and pt15
        # 4x4 and 4x4
        # during test, use larger joint pose for better visualization
        """  
        if self.sample_method == 'label':
            # sample from gt of dataset follows eg3d
            dataset_len = self.euler_sample.size()[0]
            # jointly sample joint12 and 15
            rand_id = torch.randint(0, dataset_len, (n, ))
            joint12 = self.euler_sample[rand_id, 12, :].pin_memory().unsqueeze(1).to(device)
            joint15 = self.euler_sample[rand_id, 15, :].pin_memory().unsqueeze(1).to(device)

            # jointly sample joint13 and 16
            rand_id = torch.randint(0, dataset_len, (n, ))
            joint13 = self.euler_sample[rand_id, 13, :].pin_memory().unsqueeze(1).to(device)
            joint16 = self.euler_sample[rand_id, 16, :].pin_memory().unsqueeze(1).to(device)
            
            # jointly sample joint14 and 17
            rand_id = torch.randint(0, dataset_len, (n, ))
            joint14 = self.euler_sample[rand_id, 14, :].pin_memory().unsqueeze(1).to(device)
            joint17 = self.euler_sample[rand_id, 17, :].pin_memory().unsqueeze(1).to(device)

        sample_pose = torch.cat([joint12, joint13, joint14, joint15, joint16, joint17], dim=1)
        return sample_pose


class CameraJointSampler(torch.nn.Module):
    """
    sampling camera pose and joint pose together
    """
    def __init__(self, training_dataset):
        super().__init__()
        self.training_dataset = training_dataset
        self.camera_pose_joint_sample = torch.from_numpy(training_dataset.camera_pose_joint_sample)
        self.dataset_len = self.camera_pose_joint_sample.size()[0]

    def forward(self, n, device):
        rand_id = torch.randint(0, self.dataset_len, (n, ))
        camera_pose_and_joint = self.camera_pose_joint_sample[rand_id, :].pin_memory().to(device)
        P = camera_pose_and_joint[:, :2]
        given_pose = camera_pose_and_joint[:, 2:].view(n, 6, 3)
        return P, given_pose


class ShapeSampler(torch.nn.Module):
    def __init__(self, training_dataset):
        super().__init__()
        """
        func: sample a shape parameters in the dictionary
        
        """
        self.shape_param_human = torch.from_numpy(training_dataset.shape_param_sample)
        self.zero_param_dict = torch.zeros((41000, 10))
        self.shape_param_dict_face = torch.cat([self.shape_param_human, self.zero_param_dict], dim=0)

    def forward(self, n, device):
        # sample batch * n 
        rand_id = torch.randint(0, self.shape_param_human.size()[0], (n, ))
        sample_shape = self.shape_param_human[rand_id, :].pin_memory().to(device)
        return sample_shape
    
    def forward_face(self, n, device):
        rand_id = torch.randint(0, self.shape_param_dict_face.size()[0], (n, ))
        sample_shape = self.shape_param_dict_face[rand_id, :].pin_memory().to(device)
        return sample_shape


class IDShapeSampler(torch.nn.Module):
    def __init__(self, training_dataset_face, training_dataset_body):
        super().__init__()
        """
        func: sample a id and shape from the dictionary
        
        """
        self.id_face_sampler = torch.from_numpy(training_dataset_face.id_sample)
        self.id_shape_sampler_body = torch.from_numpy(training_dataset_body.id_shape_sample)

        self.dataset_len_face = self.id_face_sampler.size()[0]
        self.dataset_len_body = self.id_shape_sampler_body.size()[0]

    def forward_face(self, n, device):
        rand_id = torch.randint(0, self.dataset_len_face, (n, ))
        id = self.id_face_sampler[rand_id, :].pin_memory().to(device)
        return id

    def forward_body(self, n, device):
        rand_id = torch.randint(0, self.dataset_len_body, (n, ))
        id_shape = self.id_shape_sampler_body[rand_id, :].pin_memory().to(device)
        id = id_shape[:, :-10]
        shape = id_shape[:, -10:]
        return id, shape


class IDShapeSamplerTest(torch.nn.Module):
    def __init__(self, training_dataset_body):
        super().__init__()
        """
        func: sample a id and shape from the dictionary
        
        """
        self.id_shape_sampler_body = torch.from_numpy(training_dataset_body.id_shape_sample)
        self.dataset_len_body = self.id_shape_sampler_body.size()[0]

    def forward_face(self, n, device):
        rand_id = torch.randint(0, self.dataset_len_face, (n, ))
        id = self.id_face_sampler[rand_id, :].pin_memory().to(device)
        return id

    def forward_body(self, n, device):
        rand_id = torch.randint(0, self.dataset_len_body, (n, ))
        id_shape = self.id_shape_sampler_body[rand_id, :].pin_memory().to(device)
        id = id_shape[:, :-10]
        shape = id_shape[:, -10:]
        return id, shape


class ExpressionSampler(torch.nn.Module):
    def __init__(self, training_dataset_face, training_dataset_body):
        super().__init__()
        """
        func: sample a id and shape from the dictionary
        """
        self.expression_face = torch.from_numpy(training_dataset_face.expression_sample)
        self.expression_body = torch.from_numpy(training_dataset_body.expression_sample)
        self.dataset_len_face = self.expression_face.size()[0]
        self.dataset_len_body = self.expression_body.size()[0]


    def forward_face(self, n, device):
        rand_id = torch.randint(0, self.dataset_len_face, (n, ))
        exp = self.expression_face[rand_id, :].pin_memory().to(device)
        return exp 


    def forward_body(self, n, device):
        rand_id = torch.randint(0, self.dataset_len_body, (n, ))
        exp = self.expression_body[rand_id, :].pin_memory().to(device)
        return exp 


class BinaryMaskSampler(torch.nn.Module):
    def __init__(self, training_dataset):
        super().__init__()
        """
        func: sample a binary mask from the face dataset to remove the effect of reflection padding
        
        """
        self.binary_mask_param = torch.from_numpy(training_dataset.all_masks)

    def forward(self, n, device):
        # sample batch * n 
        rand_id = torch.randint(0, self.binary_mask_param.size()[0], (n, ))
        sample_binary_mask = self.binary_mask_param[rand_id, :].pin_memory().to(device).permute(0, 3, 1, 2) # BHWC->BCHW
        sample_binary_mask = sample_binary_mask / 255.0
        return sample_binary_mask