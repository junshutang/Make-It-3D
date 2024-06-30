import sys

import jittor as jt
import jittor.nn as nn
import torch
from jittor import Function as F

from activation import trunc_exp
from .renderer import NeRFRenderer

import numpy as np
from encoding import get_encoder

from JNeRF.python.jnerf.models.position_encoders.hash_encoder import HashEncoder
# from freq_encoder import FrequencyEncoder
from .utils import safe_normalize


def nan_to_num(
        a,
        nan=0.0,
        posinf=None,
        neginf=None,
):
    assert isinstance(a, jt.Var)

    if a.dtype is bool or a.dtype is int:
        return a.clone()

    if nan is None:
        nan = 0.0

    if posinf is None:
        posinf = jt.misc.finfo(a.dtype).max
    if neginf is None:
        neginf = jt.misc.finfo(a.dtype).min
    result = jt.where(jt.isnan(a), nan, a)  # type: ignore[call-overload]

    result = jt.where(jt.isneginf(result), neginf, result)  # type: ignore[call-overload]
    result = jt.where(jt.isposinf(result), posinf, result)  # type: ignore[call-overload]
    return result


# TODO: not sure about the details...

class MLP(nn.Module):
    def __init__(self, dim_in, dim_out, dim_hidden, num_layers, bias=True):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.dim_hidden = dim_hidden
        self.num_layers = num_layers

        net = []
        for l in range(num_layers):
            net.append(nn.Linear(self.dim_in if l == 0 else self.dim_hidden,
                                 self.dim_out if l == num_layers - 1 else self.dim_hidden, bias=bias))

        self.net = nn.Sequential(net)

    def execute(self, x):

        for l in range(self.num_layers):
            x = self.net[l](x)
            if l != self.num_layers - 1:
                x = jt.nn.relu(x)

        return x


class NeRFNetwork(NeRFRenderer):
    def __init__(self,
                 opt,
                 bg_color=None,
                 num_layers=3,  # 5 in paper
                 hidden_dim=64,  # 128 in paper
                 num_layers_bg=2,  # 3 in paper
                 hidden_dim_bg=64,  # 64 in paper
                 # encoding='hashgrid', # pure pyjt
                 ):

        super().__init__(opt)

        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        # self.encoder, self.in_dim = get_encoder(encoding, input_dim=3, multires=6)
        self.encoder = HashEncoder()
        self.sigma_net = MLP(self.encoder.out_dim, 4, hidden_dim, num_layers, bias=True)

        # background network
        if bg_color is not None:
            bg64 = jt.nn.interpolate(
                bg_color,
                size=64,
                mode='bicubic',
                align_corners=True,
            )
            bg64 = bg64.permute(0, 2, 3, 1).view(-1, 3)
            self._bg_color = jt.Var(bg64)
        else:
            self._bg_color = jt.Var((4096, 3))
        if self.bg_radius > 0:
            self.num_layers_bg = num_layers_bg
            self.hidden_dim_bg = hidden_dim_bg
            # self.encoder_bg, self.in_dim_bg = get_encoder(encoding, input_dim=3, multires=4)
            self.encoder_bg = FrequencyEncoder()

            self.bg_net = MLP(self.in_dim_bg, 3, hidden_dim_bg, num_layers_bg, bias=True)

        else:
            self.bg_net = None

    def gaussian(self, x):
        # x: [B, N, 3]

        d = (x ** 2).sum(-1)
        g = self.opt.blob_density * jt.exp(- d / (2 * self.opt.blob_radius ** 2))

        return g

    def common_forward(self, x, is_grad=True):
        # x: [N, 3], in [-bound, bound]
        # sigma
        h = (x + self.bound) / (2 * self.bound)

        h = self.encoder(h, is_grad)

        h = self.sigma_net(h)  # syh: 此处似乎正常

        sigma = trunc_exp(h[..., 0] + self.gaussian(x))

        albedo = jt.sigmoid(h[..., 1:])

        return sigma, albedo

    # ref: https://github.com/zhaofuq/Instant-NSR/blob/main/nerf/network_sdf.py#L192
    def finite_difference_normal(self, x, epsilon=1e-2):
        # x: [N, 3]
        dx_pos, _ = self.common_forward((x + jt.array([[epsilon, 0.00, 0.00]])).clamp(-self.bound, self.bound))
        dx_neg, _ = self.common_forward((x + jt.array([[-epsilon, 0.00, 0.00]])).clamp(-self.bound, self.bound))
        dy_pos, _ = self.common_forward((x + jt.array([[0.00, epsilon, 0.00]])).clamp(-self.bound, self.bound))
        dy_neg, _ = self.common_forward((x + jt.array([[0.00, -epsilon, 0.00]])).clamp(-self.bound, self.bound))
        dz_pos, _ = self.common_forward((x + jt.array([[0.00, 0.00, epsilon]])).clamp(-self.bound, self.bound))
        dz_neg, _ = self.common_forward((x + jt.array([[0.00, 0.00, -epsilon]])).clamp(-self.bound, self.bound))

        normal = jt.stack([
            0.5 * (dx_pos - dx_neg) / epsilon,
            0.5 * (dy_pos - dy_neg) / epsilon,
            0.5 * (dz_pos - dz_neg) / epsilon
        ], dim=-1)

        return -normal

    def normal(self, x):


        normal = self.finite_difference_normal(x)
        normal = safe_normalize(normal)
        normal = nan_to_num(normal)

        return normal

    def execute(self, x, d, l=None, ratio=1, shading='albedo', is_test=False):
        # x: [N, 3], in [-bound, bound]
        # d: [N, 3], view direction, nomalized in [-1, 1]
        # l: [3], plane light direction, nomalized in [-1, 1]
        # ratio: scalar, ambient ratio, 1 == no shading (albedo only), 0 == only shading (textureless)
        # optimizer = jt.optim.Adam(self.encoder.parameters(), lr=0.5)
        if shading == 'albedo':  # syh: normal
            # normal = self.normal(x)
            if is_test:
                normal = self.normal(x)
            else:
                normal = None
            sigma, albedo = self.common_forward(x)


            color = albedo


        else:
            # query normal
            if is_test:
                normal = self.normal(x)
            else:
                normal = None
            sigma, albedo = self.common_forward(x)


            if normal.shape[0] < 1e6:
                lambertian = ratio + (1 - ratio) * (normal @ l).clamp(min=0.1)  # [N,]
                if shading == 'textureless':
                    color = lambertian.unsqueeze(-1).repeat(1, 3)
                elif shading == 'normal':
                    color = (normal + 1) / 2
                else:  # 'lambertian'
                    color = albedo * lambertian.unsqueeze(-1)
            else:
                color = albedo

        return sigma, color, normal

    def density(self, x, is_grad=True):
        # x: [N, 3], in [-bound, bound]
        # print(x)
        # print("WXZ TEST DENSITY!=====")
        sigma, albedo = self.common_forward(x, is_grad)  # syh: 这个不用梯度就行
        # print(sigma)
        return {
            'sigma': sigma,
            'albedo': albedo,
        }

    def background(self, d):

        h = self.encoder_bg(d)  # [N, C]

        h = self.bg_net(h)

        # sigmoid activation for rgb
        rgbs = jt.sigmoid(h)

        return rgbs

    # optimizer utils
    def get_params(self, lr):

        params = [
            {'params': self.encoder.parameters(), 'lr': lr * 10},
            {'params': self.sigma_net.parameters(), 'lr': lr},
            # {'params': [self.bg_color], 'lr': lr}
        ]

        if self.bg_radius > 0:
            params.append({'params': self.encoder_bg.parameters(), 'lr': lr * 10})
            params.append({'params': self.bg_net.parameters(), 'lr': lr})

        return params
