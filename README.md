# AniPortraitGAN: Animatable 3D Portrait Generation from 2D Image Collections

This is a pytorch implementation of the following paper:

Yue Wu*, Sicheng Xu*, Jianfeng Xiang, Fangyun Wei, Qifeng Chen, Jiaolong Yang, Xin Tong. **AniPortraitGAN: Animatable 3D Portrait Generation from 2D Image Collections**, SIGGRAPH Asia 2023.

### [Project page](https://yuewuhkust.github.io/AniPortraitGAN/) | [Paper](https://arxiv.org/abs/2309.02186) | [Video](https://www.youtube.com/watch?v=AMCm8kgfeqc) ###

Abstract: _Previous animatable 3D-aware GANs for human generation have primarily focused on either the human head or full body. However, head-only videos are relatively uncommon in real life, and full body generation typically does not deal with facial expression control and still has challenges in generating high-quality results. Towards applicable video avatars, we present an animatable 3D-aware GAN that generates portrait images with controllable facial expression, head pose, and shoulder movements. For this new task, we base our method on the generative radiance manifold representation and equip it with learnable facial and head-shoulder deformations. A dual-camera rendering and adversarial learning scheme is proposed to improve the quality of the generated faces, which is critical for portrait images. A pose deformation processing network is developed to generate plausible deformations for challenging regions such as long hair. Experiments show that our method, trained on unstructured 2D images, can generate diverse and high-quality 3D portraits with desired control over different properties.._


## To do
- [ ] Release inference code
- [ ] Release pretrained checkpoints


## Citation

Please cite the following paper if this work helps your research:

    @inproceedings{yue2023aniportraitgan,
    title={AniPortraitGAN: Animatable 3D Portrait Generation from 2D Image Collections},
    author={Wu, Yue and Xu, Sicheng and Xiang, Jianfeng and Wei, Fangyun and Chen, Qifeng and Yang, Jiaolong and Tong, Xin},
    booktitle={SIGGRAPH Asia 2023 Conference Proceedings},
    year={2023}
}
