import functools
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def kaiming_leaky_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        torch.nn.init.kaiming_normal_(m.weight, a=0.2, mode='fan_in', nonlinearity='leaky_relu')


def make_layer(block, n_layers):
    layers = []
    for _ in range(n_layers):
        layers.append(block())
    return nn.Sequential(*layers)


class MappingNetwork(nn.Module):
    def __init__(self, z_dim, map_hidden_dim, map_output_dim):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(z_dim, map_hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(map_hidden_dim, map_hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(map_hidden_dim, map_hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(map_hidden_dim, map_output_dim)
        )

        self.network.apply(kaiming_leaky_init)
 

    def forward(self, z):
        styles = self.network(z)
        return styles

    def get_w(self, z):
        return self.network[:-1](z)

    def from_w(self, w):
        styles = self.network[-1](w)
        return styles


class ModConv2D(nn.Conv2d):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride = 1,
        padding = 0,
        dilation = 1,
        bias=True,
    ) -> None:
        super(ModConv2D, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, 1, bias, 'zeros')

    def forward(self, x, styles):
        batchsize = x.shape[0]
        w = self.weight.unsqueeze(0) # [NOIkk]
        w = w * styles.reshape(batchsize, 1, -1, 1, 1) # [NOIkk]
        dcoefs = (w.square().sum(dim=[2,3,4]) + 1e-8).rsqrt() # [NO]
        w = w * dcoefs.reshape(batchsize, -1, 1, 1, 1) # [NOIkk]
        x = x.reshape(1, x.shape[0] * x.shape[1], x.shape[2], x.shape[3])
        w = w.reshape(batchsize * self.out_channels, self.in_channels, *self.kernel_size)
        b = self.bias.unsqueeze(0).expand(batchsize, -1).reshape(-1) if self.bias is not None else None
        return F.conv2d(x, w, b, self.stride, self.padding, self.dilation, batchsize).reshape(batchsize, self.out_channels, x.shape[2], x.shape[3])


class ResidualDenseBlock_5C(nn.Module):
    def __init__(self, nf=64, gc=32, bias=True):
        super(ResidualDenseBlock_5C, self).__init__()
        # gc: growth channel, i.e. intermediate channels
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        # initialization
        # mutil.initialize_weights([self.conv1, self.conv2, self.conv3, self.conv4, self.conv5], 0.1)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class RRDB(nn.Module):
    '''Residual in Residual Dense Block'''

    def __init__(self, nf, gc=32):
        super(RRDB, self).__init__()
        self.RDB1 = ResidualDenseBlock_5C(nf, gc)
        self.RDB2 = ResidualDenseBlock_5C(nf, gc)
        self.RDB3 = ResidualDenseBlock_5C(nf, gc)

    def forward(self, x):
        out = self.RDB1(x)
        out = self.RDB2(out)
        out = self.RDB3(out)
        return out * 0.2 + x


class StyleRRDBNet(nn.Module):
    def __init__(self, in_nc, out_nc, w_dim, nf, nb, gc=32, scale_factor=4, up_channels=None, up_sample_ks=3, to_rgb_ks=3, use_mapping_network=False, use_pixel_shuffle=True):
        super(StyleRRDBNet, self).__init__()
        self.scale_factor = scale_factor
        self.use_pixel_shuffle = use_pixel_shuffle
        RRDB_block_f = functools.partial(RRDB, nf=nf, gc=gc)

        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)
        self.RRDB_trunk = make_layer(RRDB_block_f, nb)
        self.trunk_conv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        #### upsampling
        self.num_upconvs = int(np.log2(scale_factor))
        if up_channels is None:
            up_channels = [nf] * 2 + [nf // 2] * (self.num_upconvs - 2)
        self.upconvs = nn.ModuleList([])
        style_dim = 0
        if self.use_pixel_shuffle:
            for i in range(self.num_upconvs):
                if i == 0:
                    self.upconvs.append(ModConv2D(nf, up_channels[0] * 4, 3, 1, 1, bias=True))
                    style_dim += nf
                else:
                    self.upconvs.append(ModConv2D(up_channels[i - 1], up_channels[i] * 4, up_sample_ks, 1, (up_sample_ks-1)//2, bias=True))
                    style_dim += up_channels[i - 1]
        else:
            for i in range(self.num_upconvs):
                if i == 0:
                    self.upconvs.append(ModConv2D(nf, up_channels[0], 3, 1, 1, bias=True))
                    style_dim += nf
                else:
                    self.upconvs.append(ModConv2D(up_channels[i - 1], up_channels[i], up_sample_ks, 1, (up_sample_ks-1)//2, bias=True))
                    style_dim += up_channels[i - 1]
        self.HRconv = ModConv2D(up_channels[-1], up_channels[-1], up_sample_ks, 1, (up_sample_ks-1)//2, bias=True)
        self.conv_last = ModConv2D(up_channels[-1], out_nc, to_rgb_ks, 1, (to_rgb_ks-1)//2, bias=True)
        style_dim += 2 * up_channels[-1]

        self.use_mapping_network = use_mapping_network
        self.avg_w = None
        self.avg_style = None
        if self.use_mapping_network:
            self.mapping_network = MappingNetwork(w_dim, 256, style_dim)
        else:
            self.affine = nn.Linear(w_dim, style_dim)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def get_avg_w(self, z):
        if self.use_mapping_network:
            with torch.no_grad():
                ws = self.mapping_network.get_w(z)
            self.avg_w = ws.mean(0, keepdim=True)
            self.avg_style = self.mapping_network.from_w(self.avg_w)
        else:
            self.avg_w = z.mean(0, keepdim=True)
            self.avg_style = self.affine(self.avg_w)
        return self.avg_style

    def forward(self, x, w, truncation_psi=1):
        if self.use_mapping_network:
            style = self.mapping_network(w)
        else:
            style = self.affine(w)
        if truncation_psi < 1:
            style = self.avg_style.lerp(style, truncation_psi)
        return self.forward_with_style(x, style)

    def forward_with_style(self, x, style):
        layer_num = x.shape[0] // style.shape[0]
        style = style.unsqueeze(1).expand(-1, layer_num, -1).reshape(x.shape[0], -1)

        fea = self.conv_first(x)
        trunk = self.trunk_conv(self.RRDB_trunk(fea))
        fea = fea + trunk

        start = 0
        for i in range(self.num_upconvs):
            if self.use_pixel_shuffle:
                fea = self.lrelu(F.pixel_shuffle(self.upconvs[i](fea, style[..., start:start+self.upconvs[i].in_channels]), 2))
            else:
                fea = F.interpolate(self.lrelu(self.upconvs[i](fea, style[..., start:start+self.upconvs[i].in_channels])), scale_factor=2, mode='bilinear')
            start += self.upconvs[i].in_channels
        fea = self.lrelu(self.HRconv(fea, style[..., start:start+self.HRconv.in_channels]))
        start += self.HRconv.in_channels
        out = self.conv_last(fea, style[..., start:start+self.conv_last.in_channels])

        assert out.shape[-1] == x.shape[-1] * self.scale_factor

        return out