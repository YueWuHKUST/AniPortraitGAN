SSHQ512_inf = {
    'generator': {
        'class': 'ManifoldSRGenerator3d',
        'kwargs':{
            'z_dim': 256,
            'hidden_dim': 256, 
            'hidden_dim_sample': 128,
            'feature_dim': 32,
            'hr_img_size': 512,
            'lr_img_size': 128,
            'clamp_mode': 'softplus',
            'levels_start': 23,
            'levels_end': 8,
            'num_samples': 64,
            'num_manifolds': 24,
            'last_back': False,
            'white_back': True,
            'sr_net_kwargs': {
                'w_dim': 160,
                'nf': 64, 
                'nb': 4,
                'gc': 32,
                'up_channels': [64, 32],
                'to_rgb_ks': 1,
            }
        },
    }
}
