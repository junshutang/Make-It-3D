import torch
import torch.nn as nn
import torch.nn.functional as F

from activation import trunc_exp
from .renderer import NeRFRenderer
from encoding import get_encoder

import numpy as np
import tinycudann as tcnn
from .utils import safe_normalize

class MLP(nn.Module):
    def __init__(self, dim_in, dim_out, dim_hidden, num_layers, bias=True):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.dim_hidden = dim_hidden
        self.num_layers = num_layers

        net = []
        for l in range(num_layers):
            net.append(nn.Linear(self.dim_in if l == 0 else self.dim_hidden, self.dim_out if l == num_layers - 1 else self.dim_hidden, bias=bias))

        self.net = nn.ModuleList(net)
    
    def forward(self, x):
        for l in range(self.num_layers):
            x = self.net[l](x)
            if l != self.num_layers - 1:
                x = F.relu(x, inplace=True)
        return x




class NeRFNetwork(NeRFRenderer):
    def __init__(self, 
                 opt,
                 bg_color=None,
                 num_layers=3,
                 hidden_dim=64,
                 num_layers_bg=2,
                 hidden_dim_bg=64,
                 ):
        
        super().__init__(opt)

        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        per_level_scale = np.exp2(np.log2(2048 * self.bound / 16) / (16 - 1))

        self.encoder = tcnn.Encoding(
            n_input_dims=3,
            encoding_config={
                "otype": "HashGrid",
                "n_levels": 16,
                "n_features_per_level": 2,
                "log2_hashmap_size": 19,
                "base_resolution": 16,
                "per_level_scale": per_level_scale,
            },
            dtype=torch.float32
        )

        self.sigma_net = MLP(32, 4, hidden_dim, num_layers, bias=True)
        if bg_color is not None:
            bg64 = torch.nn.functional.interpolate(
                        bg_color,
                        size=64,
                        mode="bicubic",
                        align_corners=True,
                    ) # [1, 1, 512, 512] [80~150]
            bg64 = bg64.permute(0, 2, 3, 1).view(-1, 3)
            self.bg_color = torch.tensor(bg64, requires_grad=False, device=self.device)
        else:
            self.bg_color = torch.rand((4096, 3), requires_grad=False, device=self.device)
        # background network
        if self.bg_radius > 0:
            self.num_layers_bg = num_layers_bg   
            self.hidden_dim_bg = hidden_dim_bg

            self.encoder_bg, self.in_dim_bg = get_encoder('frequency', input_dim=3)

            self.bg_net = MLP(self.in_dim_bg, 3, hidden_dim_bg, num_layers_bg, bias=True)
        else:
            self.bg_net = None
            
    def set_device(self, device):
        self.encoder.to(device)
        self.sigma_net.to(device)

    def gaussian(self, x):
        # x: [B, N, 3]
        
        d = (x ** 2).sum(-1)
        g = self.opt.blob_density * torch.exp(-d / (2 * self.opt.blob_radius ** 2))

        return g

    def common_forward(self, x):
        # x: [N, 3], in [-bound, bound]

        # sigma
        h = (x + self.bound) / (2 * self.bound) # to [0, 1]
        h = self.encoder(h)
        h = self.sigma_net(h)
        sigma = trunc_exp(h[..., 0] + self.gaussian(x))
        albedo = torch.sigmoid(h[..., 1:])

        return sigma, albedo

    # ref: https://github.com/zhaofuq/Instant-NSR/blob/main/nerf/network_sdf.py#L192
    def finite_difference_normal(self, x, epsilon=1e-2):
        # x: [N, 3]
        dx_pos, _ = self.common_forward((x + torch.tensor([[epsilon, 0.00, 0.00]], device=x.device)).clamp(-self.bound, self.bound))
        dx_neg, _ = self.common_forward((x + torch.tensor([[-epsilon, 0.00, 0.00]], device=x.device)).clamp(-self.bound, self.bound))
        dy_pos, _ = self.common_forward((x + torch.tensor([[0.00, epsilon, 0.00]], device=x.device)).clamp(-self.bound, self.bound))
        dy_neg, _ = self.common_forward((x + torch.tensor([[0.00, -epsilon, 0.00]], device=x.device)).clamp(-self.bound, self.bound))
        dz_pos, _ = self.common_forward((x + torch.tensor([[0.00, 0.00, epsilon]], device=x.device)).clamp(-self.bound, self.bound))
        dz_neg, _ = self.common_forward((x + torch.tensor([[0.00, 0.00, -epsilon]], device=x.device)).clamp(-self.bound, self.bound))
        
        normal = torch.stack([
            0.5 * (dx_pos - dx_neg) / epsilon, 
            0.5 * (dy_pos - dy_neg) / epsilon, 
            0.5 * (dz_pos - dz_neg) / epsilon
        ], dim=-1)

        return -normal
    
    def normal(self, x):
    
        normal = self.finite_difference_normal(x)
        normal = safe_normalize(normal)
        normal = torch.nan_to_num(normal)

        return normal
    
    def forward(self, x, d, l=None, ratio=1, shading='albedo'):
        # x: [N, 3], in [-bound, bound]
        # d: [N, 3], view direction, nomalized in [-1, 1]
        # l: [3], plane light direction, nomalized in [-1, 1]
        # ratio: scalar, ambient ratio, 1 == no shading (albedo only)

        if shading == 'albedo':
            # no need to query normal
            sigma, albedo = self.common_forward(x)
            normal = self.normal(x)
            color = albedo
        
        else:
            # query normal
            # has_grad = torch.is_grad_enabled()

            sigma, albedo = self.common_forward(x)
            normal = self.normal(x)
            # lambertian shading
            if normal.shape[0] < 1e6:
                lambertian = ratio + (1 - ratio) * (normal @ l).clamp(min=0.1) # [N,]
                if shading == 'textureless':
                    color = lambertian.unsqueeze(-1).repeat(1, 3)
                elif shading == 'normal':
                    color = (normal + 1) / 2
                else: # 'lambertian'
                    color = albedo * lambertian.unsqueeze(-1)
            else:
                color = albedo
            
        return sigma, color, normal

      
    def density(self, x):
        # x: [N, 3], in [-bound, bound]
        sigma, albedo = self.common_forward(x)
        
        return {
            'sigma': sigma,
            'albedo': albedo,
        }

    def background(self, d):
        # x: [N, 2], in [-1, 1]

        h = self.encoder_bg(d) # [N, C]
        
        h = self.bg_net(h)

        # sigmoid activation for rgb
        rgbs = torch.sigmoid(h)

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