import os
import glob
import tqdm
import math
import imageio
import random
import warnings
import tensorboardX

import numpy as np
import pandas as pd

import time
from datetime import datetime

import cv2
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image

import trimesh
import mcubes
from rich.console import Console
from torch_ema import ExponentialMovingAverage

import clip
import torchvision.transforms as T
from torchmetrics import PearsonCorrCoef

from packaging import version as pver

def custom_meshgrid(*args):
    # ref: https://pytorch.org/docs/stable/generated/torch.meshgrid.html?highlight=meshgrid#torch.meshgrid
    if pver.parse(torch.__version__) < pver.parse('1.10'):
        return torch.meshgrid(*args)
    else:
        return torch.meshgrid(*args, indexing='ij')

def safe_normalize(x, eps=1e-20):
    return x / torch.sqrt(torch.clamp(torch.sum(x * x, -1, keepdim=True), min=eps, max=1e32))

@torch.cuda.amp.autocast(enabled=False)
def get_rays(poses, intrinsics, H, W, N=-1, error_map=None):
    ''' get rays
    Args:
        poses: [B, 4, 4], cam2world
        intrinsics: [4]
        H, W, N: int
        error_map: [B, 128 * 128], sample probability based on training error
    Returns:
        rays_o, rays_d: [B, N, 3]
        inds: [B, N]
    '''

    device = poses.device
    B = poses.shape[0]
    fx, fy, cx, cy = intrinsics

    i, j = custom_meshgrid(torch.linspace(0, W-1, W, device=device), torch.linspace(0, H-1, H, device=device))
    i = i.t().reshape([1, H*W]).expand([B, H*W]) + 0.5
    j = j.t().reshape([1, H*W]).expand([B, H*W]) + 0.5

    results = {}

    if N > 0:
        N = min(N, H*W)

        if error_map is None:
            inds = torch.randint(0, H*W, size=[N], device=device) # may duplicate
            inds = inds.expand([B, N])
        else:

            # weighted sample on a low-reso grid
            inds_coarse = torch.multinomial(error_map.to(device), N, replacement=False) # [B, N], but in [0, 128*128)

            # map to the original resolution with random perturb.
            inds_x, inds_y = inds_coarse // 128, inds_coarse % 128 # `//` will throw a warning in torch 1.10... anyway.
            sx, sy = H / 128, W / 128
            inds_x = (inds_x * sx + torch.rand(B, N, device=device) * sx).long().clamp(max=H - 1)
            inds_y = (inds_y * sy + torch.rand(B, N, device=device) * sy).long().clamp(max=W - 1)
            inds = inds_x * W + inds_y

            results['inds_coarse'] = inds_coarse # need this when updating error_map

        i = torch.gather(i, -1, inds)
        j = torch.gather(j, -1, inds)

        results['inds'] = inds

    else:
        inds = torch.arange(H*W, device=device).expand([B, H*W])

    zs = torch.ones_like(i)
    xs = (i - cx) / fx * zs
    ys = (j - cy) / fy * zs
    directions = torch.stack((xs, ys, zs), dim=-1)
    scale = 1 / directions.pow(2).sum(-1).pow(0.5)

    directions = safe_normalize(directions)
    rays_d = directions @ poses[:, :3, :3].transpose(-1, -2) # (B, N, 3)
    rays_o = poses[..., :3, 3] # [B, 3]
    rays_o = rays_o[..., None, :].expand_as(rays_d) # [B, N, 3]

    results['rays_o'] = rays_o
    results['rays_d'] = rays_d
    results['depth_scale'] = scale

    return results


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    #torch.backends.cudnn.deterministic = True
    #torch.backends.cudnn.benchmark = True


def torch_vis_2d(x, renormalize=False):
    # x: [3, H, W] or [1, H, W] or [H, W]
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    
    if isinstance(x, torch.Tensor):
        if len(x.shape) == 3:
            x = x.permute(1,2,0).squeeze()
        x = x.detach().cpu().numpy()
        
    print(f'[torch_vis_2d] {x.shape}, {x.dtype}, {x.min()} ~ {x.max()}')
    
    x = x.astype(np.float32)
    
    # renormalize
    if renormalize:
        x = (x - x.min(axis=0, keepdims=True)) / (x.max(axis=0, keepdims=True) - x.min(axis=0, keepdims=True) + 1e-8)

    plt.imshow(x)
    plt.show()

@torch.jit.script
def linear_to_srgb(x):
    return torch.where(x < 0.0031308, 12.92 * x, 1.055 * x ** 0.41666 - 0.055)


@torch.jit.script
def srgb_to_linear(x):
    return torch.where(x < 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)

def extract_fields(bound_min, bound_max, resolution, query_func):
    N = 256
    X = torch.linspace(bound_min[0], bound_max[0], resolution).split(N)
    Y = torch.linspace(bound_min[1], bound_max[1], resolution).split(N)
    Z = torch.linspace(bound_min[2], bound_max[2], resolution).split(N)

    u = np.zeros([resolution, resolution, resolution], dtype=np.float32)
    #with torch.no_grad():
    for xi, xs in enumerate(X):
        for yi, ys in enumerate(Y):
            for zi, zs in enumerate(Z):
                xx, yy, zz = torch.meshgrid(xs, ys, zs) # for torch < 1.10, should remove indexing='ij'
                pts = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)], dim=-1) # [1, N, 3]
                val = query_func(pts).reshape(len(xs), len(ys), len(zs)) # [1, N, 1] --> [x, y, z]
                u[xi * N: xi * N + len(xs), yi * N: yi * N + len(ys), zi * N: zi * N + len(zs)] = val.detach().cpu().numpy()
                del val
    return u


def extract_geometry(bound_min, bound_max, resolution, threshold, query_func, use_sdf = False):
    #print('threshold: {}'.format(threshold))
    u = extract_fields(bound_min, bound_max, resolution, query_func)
    if use_sdf:
        u = - 1.0 *u

    #print(u.mean(), u.max(), u.min(), np.percentile(u, 50))
    
    vertices, triangles = mcubes.marching_cubes(u, threshold)

    b_max_np = bound_max.detach().cpu().numpy()
    b_min_np = bound_min.detach().cpu().numpy()

    vertices = vertices / (resolution - 1.0) * (b_max_np - b_min_np)[None, :] + b_min_np[None, :]
    
    return vertices, triangles

class Trainer(object):
    def __init__(self, 
                 name, # name of this experiment
                 opt, # extra conf
                 model, # network 
                 depth_model, # depth model
                 guidance, # guidance network
                 ref_imgs, 
                 ref_depth, 
                 ref_mask,
                 ori_imgs=None,
                 criterion=None, # loss function, if None, assume inline implementation in train_step
                 optimizer=None, # optimizer
                 ema_decay=None, # if use EMA, set the decay
                 lr_scheduler=None, # scheduler
                 metrics=[], # metrics for evaluation, if None, use val_loss to measure performance, else use the first metric.
                 local_rank=0, # which GPU am I
                 world_size=1, # total num of GPUs
                 device=None, # device to use, usually setting to None is OK. (auto choose device)
                 mute=False, # whether to mute all print
                 fp16=False, # amp optimize level
                 eval_interval=1, # eval once every $ epoch
                 max_keep_ckpt=2, # max num of saved ckpts in disk
                 workspace='workspace', # workspace to save logs & ckpts
                 best_mode='min', # the smaller/larger result, the better
                 use_loss_as_metric=True, # use loss as the first metric
                 report_metric_at_train=False, # also report metrics at training
                 use_checkpoint="latest", # which ckpt to use at init time
                 use_tensorboardX=True, # whether to use tensorboard for logging
                 scheduler_update_every_step=False, # whether to call scheduler.step() after every train step
                 ):
        
        self.name = name
        self.opt = opt
        self.mute = mute
        self.metrics = metrics
        self.local_rank = local_rank
        self.world_size = world_size
        self.workspace = workspace
        self.ema_decay = ema_decay
        self.fp16 = fp16
        self.best_mode = best_mode
        self.use_loss_as_metric = use_loss_as_metric
        self.report_metric_at_train = report_metric_at_train
        self.max_keep_ckpt = max_keep_ckpt
        self.eval_interval = eval_interval
        self.use_checkpoint = use_checkpoint
        self.use_tensorboardX = use_tensorboardX
        self.time_stamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        self.scheduler_update_every_step = scheduler_update_every_step
        self.device = device if device is not None else torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')
        self.console = Console()
        self.clip_model, self.clip_preprocess = clip.load("ViT-B/16", device=self.device, jit=False)
        self.ref_imgs = ref_imgs
        self.ori_imgs = ori_imgs
        self.depth_prediction = ref_depth
        self.depth_mask = ref_mask
    
        model.to(self.device)
        if self.world_size > 1:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
        self.model = model

        depth_model.to(self.device)
        if self.world_size > 1:
            depth_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(depth_model)
            depth_model = torch.nn.parallel.DistributedDataParallel(depth_model, device_ids=[local_rank])
        self.depth_model = depth_model
        self.depth_model.eval()

        self.depth_transform = T.Compose(
        [
            T.Resize((384, 384)),
            T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
        )

        # guide model
        self.guidance = guidance

        # text prompt
        if self.guidance is not None:
            
            for p in self.guidance.parameters():
                p.requires_grad = False

            self.prepare_text_embeddings()
        
        else:
            self.text_z = None
    
        if isinstance(criterion, nn.Module):
            criterion.to(self.device)
        self.criterion = criterion
        self.pearson = PearsonCorrCoef().to(self.device)
        

        if optimizer is None:
            self.optimizer = optim.Adam(self.model.parameters(), lr=0.001, weight_decay=5e-4) # naive adam
        else:
            self.optimizer = optimizer(self.model)

        if lr_scheduler is None:
            self.lr_scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda epoch: 1) # fake scheduler
        else:
            self.lr_scheduler = lr_scheduler(self.optimizer)

        if ema_decay is not None:
            self.ema = ExponentialMovingAverage(self.model.parameters(), decay=ema_decay)
        else:
            self.ema = None

        self.scaler = torch.cuda.amp.GradScaler(enabled=self.fp16)

        # variable init
        self.epoch = 0
        self.global_step = 0
        self.local_step = 0
        self.stats = {
            "loss": [],
            "valid_loss": [],
            "results": [], # metrics[0], or valid_loss
            "checkpoints": [], # record path of saved ckpt, to automatically remove old ckpt
            "best_result": None,
        }

        self.aug = T.Compose([
            T.Resize((224, 224)),
            T.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])

        # auto fix
        if len(metrics) == 0 or self.use_loss_as_metric:
            self.best_mode = 'min'

        # workspace prepare
        self.log_ptr = None
        if self.workspace is not None:
            os.makedirs(self.workspace, exist_ok=True)        
            self.log_path = os.path.join(workspace, f"log_{self.name}.txt")
            self.log_ptr = open(self.log_path, "a+")

            self.ckpt_path = os.path.join(self.workspace, 'checkpoints')
            self.best_path = f"{self.ckpt_path}/{self.name}.pth"
            self.img_path = os.path.join(self.workspace, 'train')
            os.makedirs(self.img_path, exist_ok=True)
            
            os.makedirs(self.ckpt_path, exist_ok=True)
            
        self.log(f'[INFO] Trainer: {self.name} | {self.time_stamp} | {self.device} | {"fp16" if self.fp16 else "fp32"} | {self.workspace}')
        self.log(f'[INFO] #parameters: {sum([p.numel() for p in model.parameters() if p.requires_grad])}')

        if self.workspace is not None:
            if self.use_checkpoint == "scratch":
                self.log("[INFO] Training from scratch ...")
            elif self.use_checkpoint == "latest":
                self.log("[INFO] Loading latest checkpoint ...")
                self.load_checkpoint()
            elif self.use_checkpoint == "latest_model":
                self.log("[INFO] Loading latest checkpoint (model only)...")
                self.load_checkpoint(model_only=True)
            elif self.use_checkpoint == "best":
                if os.path.exists(self.best_path):
                    self.log("[INFO] Loading best checkpoint ...")
                    self.load_checkpoint(self.best_path)
                else:
                    self.log(f"[INFO] {self.best_path} not found, loading latest ...")
                    self.load_checkpoint()
            else: # path to ckpt
                self.log(f"[INFO] Loading {self.use_checkpoint} ...")
                self.load_checkpoint(self.use_checkpoint)

    # calculate the text embs.
    def prepare_text_embeddings(self):

        if self.opt.text is None:
            self.log(f"[WARN] text prompt is not provided.")
            self.text_z = None
            return

        self.text = []
        self.text_z = []
        self.text.append(self.opt.text)
        self.text_z.append(self.guidance.get_text_embeds([self.opt.text], [self.opt.negative]))
        
        # # white background
        # white_text = f"{self.opt.text}, white background"
        # self.text.append(white_text)
        # white_text_z = self.guidance.get_text_embeds([white_text], [self.opt.negative])
        # self.text_z.append(white_text_z)
        
        if self.opt.need_back:
            text = f"{self.opt.text}, back view"
            negative_text = f"{self.opt.negative}"
            # explicit negative dir-encoded text
            if negative_text != '': negative_text += ', '
            negative_text += "face"
            self.text.append(text)
            text_z = self.guidance.get_text_embeds([text], [negative_text])
            self.text_z.append(text_z)
        else:
            self.text.append(self.opt.text)
            self.text_z.append(self.guidance.get_text_embeds([self.opt.text], [self.opt.negative]))

        print(self.text)

    def __del__(self):
        if self.log_ptr: 
            self.log_ptr.close()


    def log(self, *args, **kwargs):
        if self.local_rank == 0:
            if not self.mute: 
                #print(*args)
                self.console.print(*args, **kwargs)
            if self.log_ptr: 
                print(*args, file=self.log_ptr)
                self.log_ptr.flush() # write immediately to file

    def img_loss(self, rgb1, rgb2):
        # l2_loss = torch.sum(torch.sqrt(1e-8 + torch.sum((rgb1 - rgb2) ** 2, dim=1, keepdims=True)))
        # return l2_loss
        l1_loss = nn.L1Loss()(rgb1, rgb2)
        return l1_loss
        
    def depth_loss(self, pearson, pred_depth, depth_gt, mask):
        # l2_loss = torch.sum(torch.sqrt(1e-8 + torch.sum((pred_depth*mask - depth_gt*mask) ** 2, dim=1, keepdims=True)))
        # return l2_loss
        pred_depth = pred_depth.squeeze()
        pred_depth = torch.nan_to_num(pred_depth)
        depth_gt = depth_gt.squeeze().reshape(-1)
        mask = mask.squeeze().reshape(-1)
        pred_depth = pred_depth.reshape(-1)
        mask = (mask==1)
        co = pearson(pred_depth[mask], depth_gt[mask])
        return 1 - co 
    
        
    def img_clip_loss(self, rgb1, rgb2):
        image_z_1 = self.clip_model.encode_image(self.aug(rgb1))
        image_z_2 = self.clip_model.encode_image(self.aug(rgb2))
        image_z_1 = image_z_1 / image_z_1.norm(dim=-1, keepdim=True) # normalize features
        image_z_2 = image_z_2 / image_z_2.norm(dim=-1, keepdim=True) # normalize features

        loss = - (image_z_1 * image_z_2).sum(-1).mean()
        return loss
    
    def img_text_clip_loss(self, rgb, prompt):
        image_z_1 = self.clip_model.encode_image(self.aug(rgb))
        image_z_1 = image_z_1 / image_z_1.norm(dim=-1, keepdim=True) # normalize features

        text = clip.tokenize(prompt).to(self.device)
        text_z = self.clip_model.encode_text(text)
        text_z = text_z / text_z.norm(dim=-1, keepdim=True)
        loss = - (image_z_1 * text_z).sum(-1).mean()
        return loss
    ### ------------------------------	

    def train_step(self, data):

        rays_o = data['rays_o'] # [B, N, 3]
        rays_d = data['rays_d'] # [B, N, 3]
        depth_scale = data['depth_scale']

        B, N = rays_o.shape[:2]
        H, W = data['H'], data['W']

        if self.global_step < self.opt.albedo_iters or data['is_front']:
            shading = 'albedo'
            ambient_ratio = 1.0
        else: 
            rand = random.random()
            if rand > 0.5: 
                shading = 'albedo'
                ambient_ratio = 1.0
            elif rand > 0.4: 
                shading = 'textureless'
                ambient_ratio = 0.1
            else: 
                shading = 'lambertian'
                ambient_ratio = 0.1

        if self.global_step % 10 == 0:
            verbose = True
        else:
            verbose = False

        ref_imgs = self.ref_imgs
        bg_color = torch.rand(3, device=self.ref_imgs.device) # [3], frame-wise random.
        bg_img = bg_color.expand(1, 512, 512, 3).permute(0, 3, 1, 2).contiguous()
        gt_rgb = ref_imgs[:, :3, :, :] * ref_imgs[:, 3:, :, :] + bg_img * (1 - ref_imgs[:, 3:, :, :])

        # _t = time.time()
        outputs = self.model.render(rays_o, rays_d, depth_scale=depth_scale, 
                            bg_color=bg_color, staged=False, perturb=True, ambient_ratio=ambient_ratio, 
                            shading=shading, force_all_rays=True, **vars(self.opt))
        
        pred_rgb = outputs['image'].reshape(B, H, W, 3).permute(0, 3, 1, 2).contiguous() # [1, 3, H, W]
        pred_depth = outputs['depth'].reshape(B, H, W, 1).permute(0, 3, 1, 2).contiguous() # [1, 1, H, W]
        pred_ws = outputs['weights_sum'].reshape(B, 1, H, W)


        if data['is_large']:
            text_z = self.text_z[1]
            text = self.text[1]
        else:
            text_z = self.text_z[0]
            text = self.text[0]
        
        if self.global_step < self.opt.diff_iters or data['is_front']:
            loss = 0
            de_imgs = None
        else:
            loss, de_imgs = self.guidance.train_step(text_z, pred_rgb, clip_model=self.clip_model, 
            ref_text=text, islarge=data['is_large'], ref_rgb=gt_rgb, guidance_scale=self.opt.guidance_scale)

        if self.opt.lambda_opacity > 0:
            loss_opacity = (pred_ws ** 2).mean()
            if data['is_large']:
                loss = loss + self.opt.lambda_opacity * loss_opacity * 10
            else:
                loss = loss + self.opt.lambda_opacity * loss_opacity

        if self.opt.lambda_entropy > 0:
            alphas = (pred_ws).clamp(1e-5, 1 - 1e-5)
            # alphas = alphas ** 2 # skewed entropy, favors 0 over 1
            loss_entropy = (- alphas * torch.log2(alphas) - (1 - alphas) * torch.log2(1 - alphas)).mean()
            if self.global_step < self.opt.diff_iters:
                loss = loss + self.opt.lambda_entropy * loss_entropy
            else:
                loss = loss + self.opt.lambda_entropy * loss_entropy * 10

        if verbose:
            print(f"loss_entropy: {loss_entropy}, loss_opacity: {loss_opacity}")

        if self.opt.lambda_orient > 0 and 'loss_orient' in outputs:
            loss_orient = outputs['loss_orient']
            loss = loss + self.opt.lambda_orient * loss_orient
            if self.global_step < self.opt.diff_iters:
                loss = loss + self.opt.lambda_orient * loss_orient
            else:
                loss = loss + self.opt.lambda_orient * loss_orient * 10

        if self.opt.lambda_smooth > 0 and 'loss_smooth' in outputs:
            loss_smooth = outputs['loss_smooth']
            loss = loss + self.opt.lambda_smooth * loss_smooth

        
        pred_rgb = F.interpolate(pred_rgb, (512, 512), mode='bilinear', align_corners=True)
        pred_depth = F.interpolate(pred_depth, (512, 512), mode='bilinear', align_corners=True)
        
        if data['is_front']:
            loss_ref = self.opt.lambda_img * self.img_loss(pred_rgb, gt_rgb)
            loss_depth = self.opt.lambda_depth * self.depth_loss(self.pearson, pred_depth, self.depth_prediction, ~self.depth_mask)
            if verbose:
                print(f"loss_depth: {loss_depth}, loss_img: {loss_ref}")
            loss_ref += loss_depth

        else:
            loss_ref = self.opt.lambda_clip * self.img_clip_loss(pred_rgb, gt_rgb) + \
                        self.opt.lambda_clip * self.img_text_clip_loss(pred_rgb, text)

        if self.global_step % 100 == 0 or self.global_step == 1:
            save_image(pred_rgb, os.path.join(self.img_path,  f'{self.global_step}.png'))
            save_image(gt_rgb, os.path.join(self.img_path,  f'{self.global_step}_gt.png'))
            save_image(pred_depth, os.path.join(self.img_path,  f'{self.global_step}_depth.png'))
            save_image(self.depth_prediction * (~self.depth_mask), os.path.join(self.img_path,  f'{self.global_step}_ref_depth_mask.png'))
            if de_imgs is not None:
                save_image(de_imgs, os.path.join(self.img_path,  f'{self.global_step}_denoise.png'))
        
        loss = loss + loss_ref   # loss_depth = 0.01 * self.opt.lambda_img * (self.img_loss(pred_depth, self.depth_prediction) + 1e-2)
        return pred_rgb, pred_ws, loss

    def eval_step(self, data):

        rays_o = data['rays_o'] # [B, N, 3]
        rays_d = data['rays_d'] # [B, N, 3]
        depth_scale = data['depth_scale']

        B, N = rays_o.shape[:2]
        H, W = data['H'], data['W']

        shading = data['shading'] if 'shading' in data else 'albedo'
        ambient_ratio = data['ambient_ratio'] if 'ambient_ratio' in data else 1.0
        light_d = data['light_d'] if 'light_d' in data else None

        outputs = self.model.render(rays_o, rays_d, staged=True, perturb=False, bg_color=None, light_d=light_d, ambient_ratio=ambient_ratio, shading=shading, force_all_rays=True, **vars(self.opt))
        pred_rgb = outputs['image'].reshape(B, H, W, 3)
        pred_depth = outputs['depth'].reshape(B, H, W)
        loss = 0.0

        return pred_rgb, pred_depth, loss

    def test_step(self, data, bg_color=None, perturb=False):  
        rays_o = data['rays_o'] # [B, N, 3]
        rays_d = data['rays_d'] # [B, N, 3]
        depth_scale = data['depth_scale']


        B, N = rays_o.shape[:2]
        H, W = data['H'], data['W']

        if bg_color is not None:
            bg_color = bg_color.to(rays_o.device)
        else:
            bg_color = torch.ones(3, device=rays_o.device) # [3]

        shading = data['shading'] if 'shading' in data else 'albedo'
        ambient_ratio = data['ambient_ratio'] if 'ambient_ratio' in data else 1.0
        light_d = data['light_d'] if 'light_d' in data else None

        outputs = self.model.render(rays_o, rays_d, depth_scale=depth_scale, staged=True, perturb=perturb, light_d=light_d, ambient_ratio=ambient_ratio, shading=shading, force_all_rays=True, bg_color=bg_color, **vars(self.opt))

        pred_rgb = outputs['image'].reshape(B, H, W, 3)
        pred_depth = outputs['depth'].reshape(B, H, W)
        pred_mask = outputs['weights_sum'].reshape(B, H, W)
        if 'normal' in outputs:
            pred_normal = outputs['normal'].reshape(B, H, W, 3)
            return pred_rgb, pred_depth, pred_mask, pred_normal
        else:
            return pred_rgb, pred_depth, pred_mask, None


    def save_mesh(self, save_path=None, resolution=128):

        if save_path is None:
            save_path = os.path.join(self.workspace, 'mesh')

        self.log(f"==> Saving mesh to {save_path}")

        os.makedirs(save_path, exist_ok=True)

        self.model.export_mesh(save_path, resolution=resolution)

        self.log(f"==> Finished saving mesh.")

    ### ------------------------------

    def train(self, train_loader, valid_loader, max_epochs):

        assert self.text_z is not None, 'Training must provide a text prompt!'
        
        if self.use_tensorboardX and self.local_rank == 0:
            self.writer = tensorboardX.SummaryWriter(os.path.join(self.workspace, "run", self.name))

        start_t = time.time()
        
        for epoch in range(self.epoch + 1, max_epochs + 1):
            self.epoch = epoch

            self.train_one_epoch(train_loader)
            if self.epoch % self.eval_interval == 0:
                self.evaluate_one_epoch(valid_loader)
                self.save_checkpoint(full=False, best=False)
                self.save_checkpoint(full=False, best=True)

        end_t = time.time()

        self.log(f"[INFO] training takes {(end_t - start_t)/ 60:.4f} minutes.")

        if self.use_tensorboardX and self.local_rank == 0:
            self.writer.close()

    def evaluate(self, loader, name=None):
        self.use_tensorboardX, use_tensorboardX = False, self.use_tensorboardX
        self.evaluate_one_epoch(loader, name)
        self.use_tensorboardX = use_tensorboardX

    def test(self, loader, save_path=None, name=None, write_video=True):

        if save_path is None:
            save_path = os.path.join(self.workspace, 'result')

        if name is None:
            name = f'{self.name}_ep{self.epoch:04d}'

        os.makedirs(save_path, exist_ok=True)
        
        self.log(f"==> Start Test, save results to {save_path}")

        pbar = tqdm.tqdm(total=len(loader) * loader.batch_size, bar_format='{percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
        self.model.eval()

        if write_video:
            all_preds = []
            all_preds_depth = []
            all_preds_normal = []

        all_poses = []
        with torch.no_grad():
            for i, data in enumerate(loader):
                with torch.cuda.amp.autocast(enabled=self.fp16):
                    preds, preds_depth, preds_mask, preds_normal = self.test_step(data)

                # print(preds_mask)
                mask = (preds_mask>0.9).int()

                mask = mask[0].detach().cpu().numpy()
                mask = (mask * 255).astype(np.uint8)

                pred = preds[0].detach().cpu().numpy()
                pred = (pred * 255).astype(np.uint8)

                if preds_normal is not None:
                    preds_normal = preds_normal[0].detach().cpu().numpy()
                    preds_normal = (preds_normal * 255).astype(np.uint8)
                
                poses = data['poses']
                pose = poses[0].detach().cpu().numpy()
                all_poses.append(pose)

                if write_video:
                    pred_depth_cpu = preds_depth[0].detach().cpu().numpy()
                    pred_depth_cpu = (pred_depth_cpu * 255.).astype(np.uint8)

                    all_preds.append(pred)
                    if preds_normal is not None:
                        all_preds_normal.append(preds_normal)
                    all_preds_depth.append(pred_depth_cpu)
                
                pred_depth = preds_depth[0].detach().cpu().numpy()
                pred_depth = (pred_depth * 1000.).astype(np.uint16)

                cv2.imwrite(os.path.join(save_path, f'{name}_{i:04d}_rgb.png'), cv2.cvtColor(pred, cv2.COLOR_RGB2BGR))
                if preds_normal is not None:
                    cv2.imwrite(os.path.join(save_path, f'{name}_{i:04d}_normal.png'), cv2.cvtColor(preds_normal, cv2.COLOR_RGB2BGR))
                cv2.imwrite(os.path.join(save_path, f'{name}_{i:04d}_depth.png'), pred_depth)
                cv2.imwrite(os.path.join(save_path, f'{name}_{i:04d}_mask.png'), mask)
                pbar.update(loader.batch_size)

        if write_video:
            all_preds = np.stack(all_preds, axis=0)
            all_preds_normal = np.stack(all_preds_normal, axis=0)
            all_preds_depth = np.stack(all_preds_depth, axis=0)
            imageio.mimwrite(os.path.join(save_path, f'{name}_normal.mp4'), all_preds_normal, fps=25, quality=8, macro_block_size=1)
            imageio.mimwrite(os.path.join(save_path, f'{name}_rgb.mp4'), all_preds, fps=25, quality=8, macro_block_size=1)
            imageio.mimwrite(os.path.join(save_path, f'{name}_depth.mp4'), all_preds_depth, fps=25, quality=8, macro_block_size=1)
        
        all_poses = np.stack(all_poses, axis=0)
        np.save(os.path.join(save_path, f'{name}_poses.npy'), all_poses)

        self.log(f"==> Finished Test.")
    
    def train_one_epoch(self, loader):
        self.log(f"==> Start Training {self.workspace} Epoch {self.epoch}, lr={self.optimizer.param_groups[0]['lr']:.6f} ...")

        total_loss = 0
        if self.local_rank == 0 and self.report_metric_at_train:
            for metric in self.metrics:
                metric.clear()

        self.model.train()

        # distributedSampler: must call set_epoch() to shuffle indices across multiple epochs
        # ref: https://pytorch.org/docs/stable/data.html
        if self.world_size > 1:
            loader.sampler.set_epoch(self.epoch)
        
        if self.local_rank == 0:
            pbar = tqdm.tqdm(total=len(loader) * loader.batch_size, bar_format='{desc}: {percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')

        self.local_step = 0

        for data in loader:
            
            # update grid every 16 steps
            if self.model.cuda_ray and self.global_step % self.opt.update_extra_interval == 0:
                with torch.cuda.amp.autocast(enabled=self.fp16):
                    self.model.update_extra_state()
                    
            self.local_step += 1
            self.global_step += 1

            self.optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=self.fp16):
                pred_rgbs, pred_ws, loss = self.train_step(data)


            self.scaler.scale(loss).backward()
            nn.utils.clip_grad_norm(self.model.parameters(), max_norm=10)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            if self.scheduler_update_every_step:
                self.lr_scheduler.step()

            loss_val = loss.item()
            total_loss += loss_val

            if self.local_rank == 0:

                if self.use_tensorboardX:
                    self.writer.add_scalar("train/loss", loss_val, self.global_step)
                    self.writer.add_scalar("train/lr", self.optimizer.param_groups[0]['lr'], self.global_step)

                if self.scheduler_update_every_step:
                    pbar.set_description(f"loss={loss_val:.4f} ({total_loss/self.local_step:.4f}), lr={self.optimizer.param_groups[0]['lr']:.6f}")
                else:
                    pbar.set_description(f"loss={loss_val:.4f} ({total_loss/self.local_step:.4f})")
                pbar.update(loader.batch_size)

        if self.ema is not None:
            self.ema.update()

        average_loss = total_loss / self.local_step
        self.stats["loss"].append(average_loss)

        if self.local_rank == 0:
            pbar.close()
            if self.report_metric_at_train:
                for metric in self.metrics:
                    self.log(metric.report(), style="red")
                    if self.use_tensorboardX:
                        metric.write(self.writer, self.epoch, prefix="train")
                    metric.clear()

        if not self.scheduler_update_every_step:
            if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.lr_scheduler.step(average_loss)
            else:
                self.lr_scheduler.step()

        self.log(f"==> Finished Epoch {self.epoch}.")


    def evaluate_one_epoch(self, loader, name=None):
        self.log(f"++> Evaluate {self.workspace} at epoch {self.epoch} ...")

        if name is None:
            name = f'{self.name}_ep{self.epoch:04d}'

        total_loss = 0
        if self.local_rank == 0:
            for metric in self.metrics:
                metric.clear()

        self.model.eval()

        if self.ema is not None:
            self.ema.store()
            self.ema.copy_to()

        if self.local_rank == 0:
            pbar = tqdm.tqdm(total=len(loader) * loader.batch_size, bar_format='{desc}: {percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')

        self.local_step = 0

        with torch.no_grad():
            for data in loader:    
                self.local_step += 1
                with torch.cuda.amp.autocast(enabled=self.fp16):
                    preds, preds_depth, loss = self.eval_step(data)
                # save image
                save_path = os.path.join(self.workspace, 'validation', f'{name}_{self.local_step:04d}_rgb.png')
                save_path_depth = os.path.join(self.workspace, 'validation', f'{name}_{self.local_step:04d}_depth.png')

                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                pred_depth = preds_depth.reshape(1, self.opt.H, self.opt.W, 1).permute(0, 3, 1, 2).contiguous() # [1, 1, H, W]
                preds = preds.reshape(1, self.opt.H, self.opt.W, 3).permute(0, 3, 1, 2).contiguous() # [1, 1, H, W]

                save_image(pred_depth, save_path_depth)
                save_image(preds, save_path)
            
                pbar.update(loader.batch_size)

        if self.ema is not None:
            self.ema.restore()

        self.log(f"++> Evaluate epoch {self.epoch} Finished.")

    def save_checkpoint(self, name=None, full=False, best=False):

        if name is None:
            name = f'{self.name}_ep{self.epoch:04d}'

        state = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'stats': self.stats,
        }

        if self.model.cuda_ray:
            state['mean_count'] = self.model.mean_count
            state['mean_density'] = self.model.mean_density

        if full:
            state['optimizer'] = self.optimizer.state_dict()
            state['lr_scheduler'] = self.lr_scheduler.state_dict()
            state['scaler'] = self.scaler.state_dict()
            if self.ema is not None:
                state['ema'] = self.ema.state_dict()
        
        if not best:

            state['model'] = self.model.state_dict()

            file_path = f"{name}.pth"

            self.stats["checkpoints"].append(file_path)
            torch.save(state, os.path.join(self.ckpt_path, file_path))

        else:    
            if len(self.stats["results"]) > 0:
                # always save best since loss cannot reflect performance.
                if True:
                    # save ema results 
                    if self.ema is not None:
                        self.ema.store()
                        self.ema.copy_to()

                    state['model'] = self.model.state_dict()

                    if self.ema is not None:
                        self.ema.restore()
                    
                    torch.save(state, self.best_path)
            else:
                self.log(f"[WARN] no evaluated results found, skip saving best checkpoint.")
            
    def load_checkpoint(self, checkpoint=None, model_only=False):
        if checkpoint is None:
            checkpoint_list = sorted(glob.glob(f'{self.ckpt_path}/*.pth'))
            if checkpoint_list:
                checkpoint = checkpoint_list[-1]
                self.log(f"[INFO] Latest checkpoint is {checkpoint}")
            else:
                self.log("[WARN] No checkpoint found, model randomly initialized.")
                return

        checkpoint_dict = torch.load(checkpoint, map_location=self.device)
        if 'model' not in checkpoint_dict:
            self.model.load_state_dict(checkpoint_dict)
            self.log("[INFO] loaded model.")
            return

        missing_keys, unexpected_keys = self.model.load_state_dict(checkpoint_dict['model'], strict=False)
        self.log("[INFO] loaded model.")
        if len(missing_keys) > 0:
            self.log(f"[WARN] missing keys: {missing_keys}")
        if len(unexpected_keys) > 0:
            self.log(f"[WARN] unexpected keys: {unexpected_keys}")   

        if self.ema is not None and 'ema' in checkpoint_dict:
            try:
                self.ema.load_state_dict(checkpoint_dict['ema'])
                self.log("[INFO] loaded EMA.")
            except:
                self.log("[WARN] failed to loaded EMA.")

        if self.model.cuda_ray:
            if 'mean_count' in checkpoint_dict:
                self.model.mean_count = checkpoint_dict['mean_count']
            if 'mean_density' in checkpoint_dict:
                self.model.mean_density = checkpoint_dict['mean_density']
        if model_only:
            return

        self.stats = checkpoint_dict['stats']
        self.epoch = checkpoint_dict['epoch']
        self.global_step = checkpoint_dict['global_step']
        self.log(f"[INFO] load at epoch {self.epoch}, global step {self.global_step}")
        
        if self.optimizer and 'optimizer' in checkpoint_dict:
            try:
                self.optimizer.load_state_dict(checkpoint_dict['optimizer'])
                self.log("[INFO] loaded optimizer.")
            except:
                self.log("[WARN] Failed to load optimizer.")
        
        if self.lr_scheduler and 'lr_scheduler' in checkpoint_dict:
            try:
                self.lr_scheduler.load_state_dict(checkpoint_dict['lr_scheduler'])
                self.log("[INFO] loaded scheduler.")
            except:
                self.log("[WARN] Failed to load scheduler.")
        
        if self.scaler and 'scaler' in checkpoint_dict:
            try:
                self.scaler.load_state_dict(checkpoint_dict['scaler'])
                self.log("[INFO] loaded scaler.")
            except:
                self.log("[WARN] Failed to load scaler.")