import os
import cv2
import glob
import json
import tqdm
import random
import numpy as np
from scipy.spatial.transform import Slerp, Rotation
import math
import trimesh

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .utils import get_rays, safe_normalize

DIR_COLORS = np.array([
    [255, 0, 0, 255], # front
    [0, 255, 0, 255], # side
    [0, 0, 255, 255], # back
    [255, 255, 0, 255], # side
    [255, 0, 255, 255], # overhead
    [0, 255, 255, 255], # bottom
], dtype=np.uint8)

def visualize_poses(poses, dirs, size=0.1):
    # poses: [B, 4, 4], dirs: [B]

    axes = trimesh.creation.axis(axis_length=4)
    sphere = trimesh.creation.icosphere(radius=1)
    objects = [axes, sphere]

    for pose, dir in zip(poses, dirs):
        # a camera is visualized with 8 line segments.
        pos = pose[:3, 3]
        a = pos + size * pose[:3, 0] + size * pose[:3, 1] + size * pose[:3, 2]
        b = pos - size * pose[:3, 0] + size * pose[:3, 1] + size * pose[:3, 2]
        c = pos - size * pose[:3, 0] - size * pose[:3, 1] + size * pose[:3, 2]
        d = pos + size * pose[:3, 0] - size * pose[:3, 1] + size * pose[:3, 2]

        segs = np.array([[pos, a], [pos, b], [pos, c], [pos, d], [a, b], [b, c], [c, d], [d, a]])
        segs = trimesh.load_path(segs)

        # different color for different dirs
        segs.colors = DIR_COLORS[[dir]].repeat(len(segs.entities), 0)

        objects.append(segs)

    trimesh.Scene(objects).show()

def get_view_direction(thetas, phis, overhead, front):
    #                   phis [B,];          thetas: [B,]
    # front = 0         [0, front)
    # side (left) = 1   [front, 180)
    # back = 2          [180, 180+front)
    # side (right) = 3  [180+front, 360)
    # top = 4                               [0, overhead]
    # bottom = 5                            [180-overhead, 180]
    res = torch.zeros(thetas.shape[0], dtype=torch.long)
    # first determine by phis
    res[(phis < front)] = 0
    res[(phis >= front) & (phis < np.pi)] = 1
    res[(phis >= np.pi) & (phis < (np.pi + front))] = 2
    res[(phis >= (np.pi + front))] = 3
    # override by thetas
    res[thetas <= overhead] = 4
    res[thetas >= (np.pi - overhead)] = 5
    return res


def rand_poses(size, device, radius_range=[1, 1.5], theta_range=[0, 120], phi_range=[0, 360], return_dirs=False, angle_overhead=30, angle_front=60, jitter=False, uniform_sphere_rate=0.5):
    ''' generate random poses from an orbit camera
    Args:
        size: batch size of generated poses.
        device: where to allocate the output.
        radius: camera radius
        theta_range: [min, max], should be in [0, pi]
        phi_range: [min, max], should be in [0, 2 * pi]
    Return:
        poses: [size, 4, 4]
    '''

    theta_range = np.deg2rad(theta_range)
    phi_range = np.deg2rad(phi_range)
    angle_overhead = np.deg2rad(angle_overhead)
    angle_front = np.deg2rad(angle_front)
    
    radius = torch.rand(size, device=device) * (radius_range[1] - radius_range[0]) + radius_range[0]

    if random.random() < uniform_sphere_rate:
        unit_centers = F.normalize(
            torch.stack([
                (torch.rand(size, device=device) - 0.5) * 2.0,
                torch.rand(size, device=device),
                (torch.rand(size, device=device) - 0.5) * 2.0,
            ], dim=-1), p=2, dim=1
        )
        thetas = torch.acos(unit_centers[:,1])
        phis = torch.atan2(unit_centers[:,0], unit_centers[:,2])
        phis[phis < 0] += 2 * np.pi
        centers = unit_centers * radius.unsqueeze(-1)
    else:
        thetas = torch.rand(size, device=device) * (theta_range[1] - theta_range[0]) + theta_range[0]
        phis = torch.rand(size, device=device) * (phi_range[1] - phi_range[0]) + phi_range[0]

        centers = torch.stack([
            radius * torch.sin(thetas) * torch.sin(phis),
            radius * torch.cos(thetas),
            radius * torch.sin(thetas) * torch.cos(phis),
        ], dim=-1) # [B, 3]

    targets = 0

    # jitters
    if jitter:
        centers = centers + (torch.rand_like(centers) * 0.2 - 0.1)
        targets = targets + torch.randn_like(centers) * 0.2

    # lookat
    forward_vector = safe_normalize(targets - centers)
    up_vector = torch.FloatTensor([0, -1, 0]).to(device).unsqueeze(0).repeat(size, 1)
    right_vector = safe_normalize(torch.cross(forward_vector, up_vector, dim=-1))
    
    if jitter:
        up_noise = torch.randn_like(up_vector) * 0.02
    else:
        up_noise = 0

    up_vector = safe_normalize(torch.cross(right_vector, forward_vector, dim=-1) + up_noise)

    poses = torch.eye(4, dtype=torch.float, device=device).unsqueeze(0).repeat(size, 1, 1)
    poses[:, :3, :3] = torch.stack((right_vector, up_vector, forward_vector), dim=-1)
    poses[:, :3, 3] = centers

    if return_dirs:
        dirs = get_view_direction(thetas, phis, angle_overhead, angle_front)
    else:
        dirs = None
    
    return poses, dirs

def fix_poses(size, index, device, radius_range=[1, 1.5], theta_range=[0, 100], phi_range=[0, 360]):
    ''' generate random poses from an orbit camera
    Args:
        size: batch size of generated poses.
        device: where to allocate the output.
        radius: camera radius
        theta_range: [min, max], should be in [0, pi]
        phi_range: [min, max], should be in [0, 2 * pi]
    Return:
        poses: [size, 4, 4]
    '''

    theta_range = np.deg2rad(theta_range)
    phi_range = np.deg2rad(phi_range)
    
    # rand = random.random()
    if index % 4 == 0:
        radius = torch.ones(size, device=device)
        thetas = torch.ones(size, device=device) * (theta_range[1] - theta_range[0]) / 2 + theta_range[0]
        phis = torch.ones(size, device=device) * (phi_range[1] - phi_range[0]) / 2 + phi_range[0]
        # phis = torch.ones(size, device=device) * phi_range[0]
        is_front = True
        is_large = False

    else:
        radius = torch.rand(size, device=device) * (radius_range[1] - radius_range[0]) + radius_range[0]
        if phi_range[1] <= np.deg2rad(240.0) and phi_range[0] >= np.deg2rad(120.0):
            phis = torch.rand(size, device=device) * (phi_range[1] - phi_range[0]) + phi_range[0]
        else:
            rand = random.random()
            if rand > 0.85:
                phis = torch.rand(size, device=device) * (phi_range[1] - np.deg2rad(315.0)) + np.deg2rad(315.0)
            elif rand > 0.7:
                phis = torch.rand(size, device=device) * (np.deg2rad(45.0) - phi_range[0]) + phi_range[0]
            elif rand > 0.5:
                phis = torch.rand(size, device=device) * (np.deg2rad(315.0) - np.deg2rad(240.0)) + np.deg2rad(240.0)
            elif rand > 0.3:
                phis = torch.rand(size, device=device) * (np.deg2rad(120.0) - np.deg2rad(45.0)) + np.deg2rad(45.0)
            else:
                phis = torch.rand(size, device=device) * (np.deg2rad(240.0) - np.deg2rad(120.0)) + np.deg2rad(120.0)
            
        is_front = False

        rand_theta = torch.rand(size, device=device)
        thetas = rand_theta * (theta_range[1] - theta_range[0]) + theta_range[0]

    if (phis >= np.deg2rad(0) and phis <= np.deg2rad(45)) or (phis >= np.deg2rad(315) and phis <= np.deg2rad(360)):
        is_large = True
    else:
        is_large = False

    centers = torch.stack([
        radius * torch.sin(thetas) * torch.sin(phis),
        radius * torch.cos(thetas),
        radius * torch.sin(thetas) * torch.cos(phis),
    ], dim=-1) # [B, 3]

    targets = 0

    # lookat
    forward_vector = safe_normalize(targets - centers)
    up_vector = torch.FloatTensor([0, -1, 0]).to(device).unsqueeze(0).repeat(size, 1)
    right_vector = safe_normalize(torch.cross(forward_vector, up_vector, dim=-1))
    
    up_noise = 0
    up_vector = safe_normalize(torch.cross(right_vector, forward_vector, dim=-1) + up_noise)

    poses = torch.eye(4, dtype=torch.float, device=device).unsqueeze(0).repeat(size, 1, 1)
    poses[:, :3, :3] = torch.stack((right_vector, up_vector, forward_vector), dim=-1)
    poses[:, :3, 3] = centers
    
    return thetas, phis, poses, is_front, is_large


def circle_poses(device, radius=1.0, theta=60, phi=0):

    theta = np.deg2rad(theta)
    phi = np.deg2rad(phi)

    thetas = torch.FloatTensor([theta]).to(device)
    phis = torch.FloatTensor([phi]).to(device)

    centers = torch.stack([
        radius * torch.sin(thetas) * torch.sin(phis),
        radius * torch.cos(thetas),
        radius * torch.sin(thetas) * torch.cos(phis),
    ], dim=-1) # [B, 3]

    # lookat
    forward_vector = - safe_normalize(centers)
    up_vector = torch.FloatTensor([0, -1, 0]).to(device).unsqueeze(0)
    right_vector = safe_normalize(torch.cross(forward_vector, up_vector, dim=-1))
    up_vector = safe_normalize(torch.cross(right_vector, forward_vector, dim=-1))

    poses = torch.eye(4, dtype=torch.float, device=device).unsqueeze(0)
    poses[:, :3, :3] = torch.stack((right_vector, up_vector, forward_vector), dim=-1)
    poses[:, :3, 3] = centers

    return thetas, phis, poses  
    

class NeRFDataset:
    def __init__(self, opt, device, type='train', H=256, W=256, size=100):
        super().__init__()
        
        self.opt = opt
        self.device = device
        self.type = type # train, val, test

        self.H = H
        self.W = W
        self.radius_range = opt.radius_range
        self.fov = opt.fov
        self.size = size

        self.training = self.type in ['train', 'all']
        self.testing = self.type in ['test']
        self.cx = self.H / 2
        self.cy = self.W / 2

    def collate(self, index):

        B = len(index) # always 1

        if self.training:
            # random pose on the fly
            thetas, phis, poses, is_front, is_large = fix_poses(B, index[0], self.device, radius_range=self.radius_range, theta_range=self.opt.theta_range, phi_range=self.opt.phi_range)
            if is_front:
                fov = self.fov
            else:
                fov = random.random() * (self.opt.fovy_range[1] - self.opt.fovy_range[0]) + self.opt.fovy_range[0]
            
        # elif self.testing:
        #     theta = [80.0, 90.0, 100.0]
        #     length = self.size // 3
        #     i = int(index[0] // length)
        #     if i % 2  == 0:
        #         phi = ((index[0]%length)/(length -1)) * (self.opt.phi_range[1]-self.opt.phi_range[0]) + self.opt.phi_range[0]
        #     else:
        #         phi = ((index[0]%length)/(length -1)) * (self.opt.phi_range[0]-self.opt.phi_range[1]) + self.opt.phi_range[1]
        #     theta = theta[i]
            
        #     thetas, phis, poses = circle_poses(self.device, radius=1.0, theta=theta, phi=phi)
        #     is_front = False
        #     is_large = False
        #     fov = self.fov     
        else:
            phi = (index[0] / self.size) * (self.opt.phi_range[1]-self.opt.phi_range[0]) + self.opt.phi_range[0]
            thetas, phis, poses = circle_poses(self.device, radius=1.0, theta=90, phi=phi)
            is_front = False
            is_large = False
            fov = self.fov

        focal = self.H / (2 * np.tan(np.deg2rad(fov) / 2))
        intrinsics = np.array([focal, focal, self.cx, self.cy])
        # sample a low-resolution but full image for CLIP
        rays = get_rays(poses, intrinsics, self.H, self.W, -1)

        data = {
            'H': self.H,
            'W': self.W,
            'rays_o': rays['rays_o'],
            'rays_d': rays['rays_d'],
            'depth_scale': rays['depth_scale'],
            'is_front': is_front,
            'is_large': is_large,
            'poses': poses,
            'thetas': thetas, 
            'phis': phis,
        }

        return data


    def dataloader(self):
        loader = DataLoader(list(range(self.size)), batch_size=1, collate_fn=self.collate, shuffle=self.training, num_workers=0)
        return loader