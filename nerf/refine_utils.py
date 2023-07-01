import numpy as np
import cv2
import os
import torch.nn.functional as F
import imageio
import torch
import numpy as np
from pytorch3d.structures import Pointclouds
from pytorch3d.renderer import compositing
from pytorch3d.renderer import AlphaCompositor
from pytorch3d.renderer.points import rasterize_points
import cv2
import tqdm
import random
import warnings
from copy import deepcopy

warnings.filterwarnings('ignore')

def load_depth(file_name):
    import cv2
    D = cv2.imread(file_name, cv2.IMREAD_UNCHANGED)
    if not D is None and len(D.shape) > 2:
        D = D[:,:,0]
    return    D

def save_obj(file_name, v, tri = []):
    with open(file_name, 'w') as fid:
        for i in range(len(v)):
            fid.write('v %f %f %f\n' % (v[i][0],v[i][1],v[i][2]))
        for i in range(len(tri)):
            fid.write(('f'+' %d'*len(tri[i])*'\n') % tuple( \
                [tri[i][j]+1 for j in range(len(tri[i]))]))
    return    os.path.exists(file_name)

def multidepth2point(allD, alphamask, cam, c2w, npoint=1000000):
    
    v_list = []
    for i in range(allD.shape[0]):
        D = allD[i]
        x, y = np.meshgrid(np.arange(D.shape[0]), np.arange(D.shape[1]))
        cam_xyz = np.vstack(( \
            x.reshape(-1), \
            y.reshape(-1), \
            np.ones(len(D.reshape(-1)))))

        v = np.linalg.inv(cam).dot(cam_xyz).T
        v = v * np.tile(D.reshape((-1,1)), (1,3))
        v = np.matmul(v,c2w[i, :3,:3].T)+c2w[i, :3, 3:].T
        v = v[alphamask[i].reshape(-1)==1]
        v_list.append(v)
    
    v_list = np.concatenate(v_list)
    np.random.shuffle(v_list)
    v_list = v_list[:npoint]
    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(v_list[:, :3])
    # o3d.io.write_point_cloud('ori.ply', pcd)
    return v_list

def multidepth2point_mask(allD, alphamask, allimg, cam, c2w, cano_v, cano_c2w, cano_D, H, W, radius, ppp, outputdir, device, npoint=1000000):
    image_size = [H, W]
    K = cam
    cano_v = torch.tensor(cano_v).to(device).float() # N x 3
    cano_w2c = np.linalg.inv(cano_c2w)
    cano_D = torch.Tensor(cano_D).unsqueeze(0).unsqueeze(0)
    cano_color = torch.ones_like(cano_v)
    K_tensor = torch.tensor(cam).to(device).float()
    radius = float(radius) / float(image_size[0]) * 2.0
    v_list = []
    v_color_list = []
    pbar = tqdm.tqdm(range(c2w.shape[0]))
    for i in pbar:
        if i > -1:
            gt_rgb = torch.Tensor(allimg[i:i+1,...]).permute(0,3,1,2)
            w2c = np.linalg.inv(c2w[i])
            w2c_tensor = torch.tensor(w2c).to(device)
            cano_mask = render_point(cano_v, cano_color, H, W, K_tensor, w2c_tensor, image_size, radius, ppp)
            imageio.imwrite(outputdir+'/render_%s_mask.png'%(i),np.array(cano_mask[0].permute(1,2,0).detach().cpu().numpy() * 255,dtype=np.uint8))
            mask = imageio.imread(outputdir+'/render_%s_mask.png'%(i)) / 255
            os.remove(outputdir+'/render_%s_mask.png'%(i))
            kernel = np.ones(((15, 15)), np.uint8) ##11
            mask = cv2.erode(mask,kernel,iterations=1)
            mask = np.logical_or(np.logical_or((mask[:,:, 0]>0.9),(mask[:,:, 1]>0.9)), (mask[:,:, 2]>0.9))
            mask = np.logical_and(alphamask[i], ~mask)

            D = allD[i]
            x, y = np.meshgrid(np.arange(D.shape[0]), np.arange(D.shape[1]))
            cam_xyz = np.vstack(( \
                x.reshape(-1), \
                y.reshape(-1), \
                np.ones(len(D.reshape(-1)))))
            v = np.linalg.inv(cam).dot(cam_xyz).T
            v = v * np.tile(D.reshape((-1,1)), (1,3))
            # imageio.imwrite(outputdir+'/render_%s_new_mask.png'%(i),np.array(mask * 255,dtype=np.uint8))
        
            v = v[mask.reshape(-1)==1]
            
            v = np.matmul(v,c2w[i, :3,:3].T)+c2w[i, :3, 3:].T
            xy1, xyz = project(v, K, cano_w2c[:3, :4])
            xy1 = np.round(xy1).astype(np.int32)
            xy1 =torch.Tensor(xy1[None,None,...])/H*2.-1.
            xy_d=F.grid_sample(cano_D, xy1)
            xy_d = xy_d.squeeze(0).squeeze(0).permute(1,0).cpu().numpy()
            xy_mask = np.logical_and((xyz-xy_d) <= (1 / H), (xyz-xy_d) >= -0.2)
            # xy_mask = ((xyz-xy_d) >= -0.2)
            v = v[~xy_mask[:, 0], :]

            mask_cano = z_buffer(v, w2c, H, W, K)
            v = v[mask_cano,:]  #有color的点
            xy,_ = project(v ,K, w2c[:3,:4])
            xy =torch.Tensor(xy[None,None,...])/H*2.-1.
            v_color=F.grid_sample(gt_rgb,xy)
            v_color = v_color.squeeze().permute(1,0).cpu().numpy()
            
            v_list.append(v)
            v_color_list.append(v_color)

    v_list = np.concatenate(v_list)
    v_color_list = np.concatenate(v_color_list)
    if len(v_list) < npoint:
        return v_list, v_color_list
    else:
        arr = np.arange(len(v_list))
        np.random.shuffle(arr)
        arr = arr[:npoint]
        return v_list[arr], v_color_list[arr]

def depth2point(D, alphamask, c2w, gt_rgb, H, W, cam):
    K = cam
    x, y = np.meshgrid(np.arange(D.shape[1]), np.arange(D.shape[0]))
    cam_xyz = np.vstack(( \
        x.reshape(-1), \
        y.reshape(-1), \
        np.ones(len(D.reshape(-1)))))
    v = np.linalg.inv(cam).dot(cam_xyz).T 
    v = v * np.tile(D.reshape((-1,1)), (1,3))
    v = v[alphamask.reshape(-1)==1]
    v = np.matmul(v,c2w[:3,:3].T)+c2w[:3, 3:].T
    # save_mesh = trimesh.points.PointCloud(vertices = v)
    # save_mesh.export(outputdir + '/teddy_depth_fix.obj')

    w2c = np.linalg.inv(c2w)
    mask_cano = z_buffer(v, w2c, H, W, K)
    v = v[mask_cano,:]  #有color的点

    xy,_ = project(v ,K, w2c[:3,:4])
    gt_rgb = torch.Tensor(gt_rgb[None,...]).permute(0,3,1,2)
    xy =torch.Tensor(xy[None,None,...])/H*2.-1.
    v_color=F.grid_sample(gt_rgb,xy)
    v_color = v_color.squeeze().permute(1,0).cpu().numpy()
    return v, v_color
    
def project(xyz, K, RT):
    xyz = np.dot(xyz, RT[:, :3].T) + RT[:, 3:].T
    xyz = np.dot(xyz, K.T)
    xy = xyz[:, :2] / xyz[:, 2:]
    return xy,xyz[:,2:]

def safe_normalize(x, eps=1e-20):
    return x / np.sqrt(np.clip(np.sum(x * x, axis=-1), eps, 100000))

def safe_normalize_tensor(x, eps=1e-20):
    return x / torch.sqrt(torch.clamp(torch.sum(x * x, -1, keepdim=True), min=eps))

# borrow from https://github.com/jiaxinxie97/HFGI3D/blob/main/multi_views/generate_multi_views.py
def z_buffer(vertices, world2cam_pose_cano, H, W, K):
    # world to camera
    xy1,xyz = project(vertices, K, world2cam_pose_cano[:3,:4])
    xy1 = np.round(xy1).astype(np.int32)
    xy1_mask =np.logical_and(np.logical_and((xy1[:,0]>=0),(xy1[:,0]<=W-1)),np.logical_and((xy1[:,1]>=0),(xy1[:,1]<=H-1)))

    # zbuffer, find visible point
    img1 = np.zeros((H,W))
    zbuff = np.zeros((H,W,2))
    maskout =[]
    colorout = []
    for i in np.arange(xy1.shape[0]): 
        if xy1_mask[i]:
            if img1[xy1[i, 1], xy1[i, 0]]==0:
                img1[xy1[i, 1], xy1[i, 0]] = 1
                zbuff[xy1[i,1], xy1[i,0],0] = xyz[i,0]
                zbuff[xy1[i,1], xy1[i,0],1] = i
            elif xyz[i,0]<zbuff[xy1[i,1],xy1[i,0,],0]:
                zbuff[xy1[i,1],xy1[i,0],0] = xyz[i,0]
                zbuff[xy1[i,1], xy1[i,0],1] = i
            else:
                continue 
      
    for i in np.arange(xy1.shape[0]): 
        if xy1_mask[i]:
            if (xyz[i,0]-zbuff[xy1[i,1],xy1[i,0,],0])> (1./H):
                #  continue
                maskout.append(vertices[int(i):int(i)+1,:]) 
                colorout.append(np.array([0,0,0],dtype=np.float32).reshape((1,3)))
            else:
                maskout.append(vertices[int(i):int(i)+1,:]) 
                colorout.append(np.array([1,1,1],dtype=np.float32).reshape((1,3)))
        else:
            maskout.append(vertices[int(i):int(i)+1,:]) 
            colorout.append(np.array([0,0,0],dtype=np.float32).reshape((1,3)))
    maskout = np.concatenate(maskout,axis=0)
    colorout = np.concatenate(colorout,axis=0)
    vertices = maskout
    colors = colorout
    mask =((colors[:,0]+colors[:,1]+colors[:,2])!=0)
    
    return mask

def rand_poses(index, device, radius_range=[1, 1.5], theta_range=[0, 100], phi_range=[0, 360]):

    theta_range = np.deg2rad(theta_range)
    phi_range = np.deg2rad(phi_range)
    size = 1
    if index % 4 == 0:
        radius = torch.ones(size, device=device)
        thetas = torch.ones(size, device=device) * (theta_range[1] - theta_range[0]) / 2 + theta_range[0]
        phis = torch.ones(size, device=device) * (phi_range[1] - phi_range[0]) / 2 + phi_range[0]
        is_front = True
    else:
        radius = torch.rand(size, device=device) * (radius_range[1] - radius_range[0]) + radius_range[0]
        phis = torch.rand(size, device=device) * (phi_range[1] - phi_range[0]) + phi_range[0]
        thetas = torch.rand(size, device=device) * (theta_range[1] - theta_range[0]) + theta_range[0]
        is_front = False

    centers = torch.stack([
        radius * torch.sin(thetas) * torch.sin(phis),
        radius * torch.cos(thetas),
        radius * torch.sin(thetas) * torch.cos(phis),
    ], dim=-1) # [B, 3]

    targets = 0
    # lookat
    forward_vector = safe_normalize_tensor(targets - centers)
    up_vector = torch.FloatTensor([0, -1, 0]).to(device).unsqueeze(0).repeat(size, 1)
    right_vector = safe_normalize_tensor(torch.cross(forward_vector, up_vector, dim=-1))

    poses = torch.eye(4, dtype=torch.float, device=device).unsqueeze(0).repeat(size, 1, 1)
    poses[:, :3, :3] = torch.stack((right_vector, up_vector, forward_vector), dim=-1)
    poses[:, :3, 3] = centers
    
    return poses, is_front

def fix_poses(index, device, radius_range=[1, 1.5], theta_range=[0, 100], phi_range=[0, 360]):

    theta_range = np.deg2rad(theta_range)
    phi_range = np.deg2rad(phi_range)
    size = 1
    
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
    forward_vector = safe_normalize_tensor(targets - centers)
    up_vector = torch.FloatTensor([0, -1, 0]).to(device).unsqueeze(0).repeat(size, 1)
    right_vector = safe_normalize_tensor(torch.cross(forward_vector, up_vector, dim=-1))
    
    up_noise = 0
    up_vector = safe_normalize_tensor(torch.cross(right_vector, forward_vector, dim=-1) + up_noise)

    poses = torch.eye(4, dtype=torch.float, device=device).unsqueeze(0).repeat(size, 1, 1)
    poses[:, :3, :3] = torch.stack((right_vector, up_vector, forward_vector), dim=-1)
    poses[:, :3, 3] = centers

    return poses, is_front, is_large

def render_point(points_xyz_org, points_color, H, W, K, world2cam, image_size, radius, ppp, bg_feat=None, acc='alphacomposite'):
    proj_xyz = torch.matmul(points_xyz_org, world2cam[:3, :3].T) + world2cam[:3, 3]

    # Perspective projection
    proj_xyz = torch.matmul(proj_xyz, K.T)  
    proj_xyz[:, 0:2] = proj_xyz[:, 0:2] / proj_xyz[:, 2:]
    proj_xyz[:, 0] = proj_xyz[:, 0] / W * 2 - 1.0
    proj_xyz[:, 1] = proj_xyz[:, 1] / H * 2 - 1.0
    proj_xyz[:, 0] = proj_xyz[:, 0] * -1
    proj_xyz[:, 1] = proj_xyz[:, 1] * -1

    pts3D = Pointclouds(points=[proj_xyz], features=[points_color])
    points_idx, z_buf, dist = rasterize_points(
        pts3D, image_size, radius, ppp)
    dist = 0.1*dist/ pow(radius, 2)
    alphas = (
        (1 - dist.clamp(max=1, min=1e-3).pow(0.5))
        .pow(1)
        .permute(0, 3, 1, 2)
    )

    transformed_src_alphas = compositing.alpha_composite(
        points_idx.permute(0, 3, 1, 2).long(),
        alphas,
        pts3D.features_packed().permute(1,0),
    )
        
    return transformed_src_alphas

def load_views(gt_rgb, rgb_files, depth_files, mask_files, cam2world_list, H, W, K, radius, ppp, outputdir, device):
    
    # Load single-view depth
    print("###### Loading single-view depth image ######")
    ind = (len(cam2world_list)-1) // 2
    depth_file_cano = depth_files[ind]
    mask_file_cano = mask_files[ind]
    cam2world_cano = cam2world_list[ind]
    depth_cano = load_depth(depth_file_cano) / 1000.0

    mask_cano = load_depth(mask_file_cano) / 255.0
    depth_cano = cv2.resize(depth_cano,(W,H))
    mask_cano = cv2.resize(mask_cano,(W,H))
    kernel = np.ones(((11, 11)), np.uint8) ##11
    mask_cano = cv2.erode(mask_cano,kernel,iterations=2)
    mask_cano = (mask_cano==1)

    depth_blur = depth_cano * mask_cano * 255.0
    depth_blur = np.uint8(depth_blur)
    sobel = cv2.Canny(depth_blur, 30, 30)
    kernel = np.ones(((11, 11)), np.uint8)
    edge_mask_cano = cv2.dilate(sobel, kernel,iterations=1)
    # cv2.imwrite(os.path.join(outputdir, f'render_cano_sobel.png'), edge_mask_cano)

    edge_mask_cano = (edge_mask_cano==255)
    # mask_cano = np.logical_and(mask_cano, ~edge_mask_cano)
    #### depth to point cloud
    print("###### Depth to point cloud and mesh ######")
    print("###### Single view point cloud colorization ######")
    vertices_cano, vertices_color_cano = depth2point(depth_cano, mask_cano, cam2world_cano, gt_rgb, H, W, K)

    # Load multi-view depth
    print("###### Loading multi-view depth image ######")
    all_depth=[]
    all_mask=[]
    all_cam = []
    all_rgb = []
    for i in range(len(depth_files)):
        if i == ind:
            continue
        else:
            depth = load_depth(depth_files[i]) / 1000.0
            mask = load_depth(mask_files[i]) / 255.0
            rgb = imageio.imread(rgb_files[i])/255.
            
            depth = cv2.resize(depth,(W,H))
            mask = cv2.resize(mask,(W,H))

            kernel = np.ones(((11, 11)), np.uint8) ##11
            mask = cv2.erode(mask,kernel,iterations=1)
            mask = (mask==1)
            depth_blur = depth * mask * 255.0
            depth_blur = np.uint8(depth_blur)
            sobel = cv2.Canny(depth_blur, 10, 10)
            kernel = np.ones(((11, 11)), np.uint8)
            edge_mask = cv2.dilate(sobel, kernel,iterations=1)
            # cv2.imwrite(os.path.join(outputdir, f'render_{i}_sobel.png'),edge_mask)

            edge_mask = (edge_mask==255)
            rgb = cv2.resize(rgb[:,:,:3],(W,H))
            mask = np.logical_and(mask, ~edge_mask)
            c2w = cam2world_list[i]
            all_depth.append(depth)
            all_mask.append(mask)
            all_cam.append(c2w)
            all_rgb.append(rgb)
    
    all_depth = np.stack(all_depth, axis=0)
    all_mask = np.stack(all_mask, axis=0)
    all_cam = np.stack(all_cam, axis=0)
    all_rgb = np.stack(all_rgb, axis=0)
    #### depth to point cloud
    print("###### Multi depth to point cloud and mesh ######")
    vertices_novel, vertices_color_novel = multidepth2point_mask(all_depth, all_mask, all_rgb, K, all_cam, vertices_cano, cam2world_cano, depth_cano*mask_cano, H, W, radius, ppp, outputdir, device)
    
    return vertices_cano, vertices_color_cano, vertices_novel, vertices_color_novel