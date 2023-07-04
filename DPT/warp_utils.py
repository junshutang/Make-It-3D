from locale import normalize
import sys, os
import math
import cv2
import numpy as np
import torch
from einops import rearrange
from functools import reduce
from torchvision.utils import save_image
from util.io import write_depth_name
sys.path.extend([
    'pytorch3d-lite',
])

import py3d_tools as p3d


def sample_to_cv2(sample: torch.Tensor, type=np.uint8) -> np.ndarray:
    sample_f32 = rearrange(sample.squeeze().cpu().numpy(), "c h w -> h w c").astype(np.float32)
    sample_f32 = ((sample_f32 * 0.5) + 0.5).clip(0, 1)
    sample_int8 = (sample_f32 * 255)
    sample_int8 = cv2.cvtColor(sample_int8, cv2.COLOR_BGR2RGB)
    return sample_int8.astype(type)

def construct_RotationMatrixHomogenous(rotation_angles):
    assert(type(rotation_angles)==list and len(rotation_angles)==3)
    RH = np.eye(4,4)
    cv2.Rodrigues(np.array(rotation_angles), RH[0:3, 0:3])
    return RH

# https://en.wikipedia.org/wiki/Rotation_matrix
def getRotationMatrixManual(rotation_angles):
	
    rotation_angles = [np.deg2rad(x) for x in rotation_angles]
    
    phi         = rotation_angles[0] # around x
    gamma       = rotation_angles[1] # around y
    theta       = rotation_angles[2] # around z
    
    # X rotation
    Rphi        = np.eye(4,4)
    sp          = np.sin(phi)
    cp          = np.cos(phi)
    Rphi[1,1]   = cp
    Rphi[2,2]   = Rphi[1,1]
    Rphi[1,2]   = -sp
    Rphi[2,1]   = sp
    
    # Y rotation
    Rgamma        = np.eye(4,4)
    sg            = np.sin(gamma)
    cg            = np.cos(gamma)
    Rgamma[0,0]   = cg
    Rgamma[2,2]   = Rgamma[0,0]
    Rgamma[0,2]   = sg
    Rgamma[2,0]   = -sg
    
    # Z rotation (in-image-plane)
    Rtheta      = np.eye(4,4)
    st          = np.sin(theta)
    ct          = np.cos(theta)
    Rtheta[0,0] = ct
    Rtheta[1,1] = Rtheta[0,0]
    Rtheta[0,1] = -st
    Rtheta[1,0] = st
    
    R           = reduce(lambda x,y : np.matmul(x,y), [Rphi, Rgamma, Rtheta]) 
    
    return R


def getPoints_for_PerspectiveTranformEstimation(ptsIn, ptsOut, W, H, sidelength):
    
    ptsIn2D      =  ptsIn[0,:]
    ptsOut2D     =  ptsOut[0,:]
    ptsOut2Dlist =  []
    ptsIn2Dlist  =  []
    
    for i in range(0,4):
        ptsOut2Dlist.append([ptsOut2D[i,0], ptsOut2D[i,1]])
        ptsIn2Dlist.append([ptsIn2D[i,0], ptsIn2D[i,1]])
    
    pin  =  np.array(ptsIn2Dlist)   +  [W/2.,H/2.]
    pout = (np.array(ptsOut2Dlist)  +  [1.,1.]) * (0.5*sidelength)
    pin  = pin.astype(np.float32)
    pout = pout.astype(np.float32)
    
    return pin, pout

def warpMatrix(W, H, theta, phi, gamma, scale, fV):
    
    # M is to be estimated
    M          = np.eye(4, 4)
    
    fVhalf     = np.deg2rad(fV/2.)
    d          = np.sqrt(W*W+H*H)
    sideLength = scale*d/np.cos(fVhalf)
    h          = d/(2.0*np.sin(fVhalf))
    n          = h-(d/2.0);
    f          = h+(d/2.0);
    
    # Translation along Z-axis by -h
    T       = np.eye(4,4)
    T[2,3]  = -h
    
    # Rotation matrices around x,y,z
    R = getRotationMatrixManual([phi, gamma, theta])
    
    
    # Projection Matrix 
    P       = np.eye(4,4)
    P[0,0]  = 1.0/np.tan(fVhalf)
    P[1,1]  = P[0,0]
    P[2,2]  = -(f+n)/(f-n)
    P[2,3]  = -(2.0*f*n)/(f-n)
    P[3,2]  = -1.0
    
    # pythonic matrix multiplication
    F       = reduce(lambda x,y : np.matmul(x,y), [P, T, R]) 
    
    # shape should be 1,4,3 for ptsIn and ptsOut since perspectiveTransform() expects data in this way. 
    # In C++, this can be achieved by Mat ptsIn(1,4,CV_64FC3);
    ptsIn = np.array([[
                 [-W/2., H/2., 0.],[ W/2., H/2., 0.],[ W/2.,-H/2., 0.],[-W/2.,-H/2., 0.]
                 ]])
    ptsOut  = np.array(np.zeros((ptsIn.shape), dtype=ptsIn.dtype))
    ptsOut  = cv2.perspectiveTransform(ptsIn, F)
    
    ptsInPt2f, ptsOutPt2f = getPoints_for_PerspectiveTranformEstimation(ptsIn, ptsOut, W, H, sideLength)
    
    # check float32 otherwise OpenCV throws an error
    assert(ptsInPt2f.dtype  == np.float32)
    assert(ptsOutPt2f.dtype == np.float32)
    M33 = cv2.getPerspectiveTransform(ptsInPt2f,ptsOutPt2f)

    return M33, sideLength

def transform_image_3d(image_tensor, depth_tensor, rot_mat, translate, camera_origin, device):
    # adapted and optimized version of transform_image_3d from Disco Diffusion https://github.com/alembics/disco-diffusion 
    # save_image(image_tensor, "warp_input.png", normalize=True)
    prev_img_cv2 = sample_to_cv2(image_tensor, type=np.float32)
    # cv2.imwrite("warp_tmp.png", prev_img_cv2)
    w, h = prev_img_cv2.shape[1], prev_img_cv2.shape[0]

    aspect_ratio = float(w)/float(h)
    near, far, fov_deg = 0.2, 2., 50
    cam_dir = rot_mat @ ((-camera_origin).reshape(1, 3, 1))
    # translate_new = torch.tensor([translate]).to(device) - cam_dir.squeeze(-1)
    translate_new = torch.tensor([translate]).to(device)
    rot_mat_t = rot_mat.transpose(1, 2)
    persp_cam_old = p3d.FoVPerspectiveCameras(near, far, aspect_ratio, fov=fov_deg, degrees=True, device=device)
    persp_cam_new = p3d.FoVPerspectiveCameras(near, far, aspect_ratio, fov=fov_deg, degrees=True, R=rot_mat_t, T=translate_new, device=device)

    # range of [-1,1] is important to torch grid_sample's padding handling
    y,x = torch.meshgrid(torch.linspace(-1.,1.,h,dtype=torch.float32,device=device),torch.linspace(-1.,1.,w,dtype=torch.float32,device=device))
    z = torch.as_tensor(depth_tensor, dtype=torch.float32, device=device)
    # print(z)
    # print(z.min(), z.max())
    # exit()
    xyz_old_world = torch.stack((x.flatten(), y.flatten(), z.flatten()), dim=1)

    xyz_old_cam_xy = persp_cam_old.get_full_projection_transform().transform_points(xyz_old_world)[:,0:2]
    xyz_new_cam_xy = persp_cam_new.get_full_projection_transform().transform_points(xyz_old_world)[:,0:2]
    xyz_new_cam_z = persp_cam_new.get_full_projection_transform().transform_points(xyz_old_world)[:,2]

    offset_xy = xyz_new_cam_xy - xyz_old_cam_xy
    xyz_new_cam_z = xyz_new_cam_z.reshape(1, 1, w, h)
    # warped_depth_cpu = xyz_new_cam_z.cpu().numpy()
    # write_depth_name('warp_depth.png', warped_depth_cpu, bits=2)
    # exit()
    # affine_grid theta param expects a batch of 2D mats. Each is 2x3 to do rotation+translation.
    identity_2d_batch = torch.tensor([[1.,0.,0.],[0.,1.,0.]], device=device).unsqueeze(0)
    # coords_2d will have shape (N,H,W,2).. which is also what grid_sample needs.
    coords_2d = torch.nn.functional.affine_grid(identity_2d_batch, [1,1,h,w], align_corners=False)
    offset_coords_2d = coords_2d - torch.reshape(offset_xy, (h,w,2)).unsqueeze(0)

    # image_tensor = rearrange(torch.from_numpy(prev_img_cv2.astype(np.float32)), 'h w c -> c h w').to(device)
    image_tensor = image_tensor.squeeze(0)
    new_image = torch.nn.functional.grid_sample(
        image_tensor.add(1/512 - 0.0001).unsqueeze(0), 
        offset_coords_2d, 
        mode="bicubic", 
        padding_mode="zeros", 
        align_corners=True
    )
    depth_tensor = depth_tensor.squeeze(0)
    new_depth = torch.nn.functional.grid_sample(
        depth_tensor.to(torch.float32).add(1/512 - 0.0001).unsqueeze(0), 
        offset_coords_2d, 
        mode="bicubic", 
        padding_mode="zeros", 
        align_corners=True
    )


    # convert back to cv2 style numpy array
    # result = rearrange(
    #     new_image.squeeze().clamp(0,255), 
    #     'c h w -> h w c'
    # ).cpu().numpy().astype(prev_img_cv2.dtype)
    
    return new_image, xyz_new_cam_z

def transform_depth_3d(depth_tensor, rot_mat, translate, camera_origin, device):
    w, h = depth_tensor.shape[2], depth_tensor.shape[3]
    depth_tensor = depth_tensor.to(torch.float32)
    aspect_ratio = float(w)/float(h)
    near, far, fov_deg = 0.2, 2., 100
    cam_dir = rot_mat @ ((-camera_origin).reshape(1, 3, 1))
    # translate_new = torch.tensor([translate]).to(device) - cam_dir.squeeze(-1)
    translate_new = torch.tensor([translate]).to(device).to(torch.float32)
    rot_mat_t = rot_mat.transpose(1, 2)
    persp_cam_old = p3d.FoVPerspectiveCameras(near, far, aspect_ratio, fov=fov_deg, degrees=True, device=device)
    persp_cam_new = p3d.FoVPerspectiveCameras(near, far, aspect_ratio, fov=fov_deg, degrees=True, R=rot_mat_t, T=translate_new, device=device)

    # range of [-1,1] is important to torch grid_sample's padding handling
    y,x = torch.meshgrid(torch.linspace(-1.,1.,h,dtype=torch.float32,device=device),torch.linspace(-1.,1.,w,dtype=torch.float32,device=device))
    z = torch.as_tensor(depth_tensor, dtype=torch.float32, device=device)
    xyz_old_world = torch.stack((x.flatten(), y.flatten(), z.flatten()), dim=1)
    xyz_old_cam_xy = persp_cam_old.get_full_projection_transform().transform_points(xyz_old_world)[:,0:2]
    xyz_new_cam_xy = persp_cam_new.get_full_projection_transform().transform_points(xyz_old_world)[:,0:2]

    offset_xy = xyz_new_cam_xy - xyz_old_cam_xy
    # affine_grid theta param expects a batch of 2D mats. Each is 2x3 to do rotation+translation.
    identity_2d_batch = torch.tensor([[1.,0.,0.],[0.,1.,0.]], device=device).unsqueeze(0)
    # coords_2d will have shape (N,H,W,2).. which is also what grid_sample needs.
    coords_2d = torch.nn.functional.affine_grid(identity_2d_batch, [1,1,h,w], align_corners=False)
    offset_coords_2d = coords_2d - torch.reshape(offset_xy, (h,w,2)).unsqueeze(0)

    depth_tensor = depth_tensor.squeeze(0)
    new_depth = torch.nn.functional.grid_sample(
        depth_tensor.to(torch.float32).add(1/512 - 0.0001).unsqueeze(0), 
        offset_coords_2d, 
        mode="bicubic", 
        padding_mode="border", 
        align_corners=True
    )

    return new_depth

def anim_warp_3d(prev_img_cv2, depth, rotate_dict, camera_origin, device):
    TRANSLATION_SCALE = 1.0/200.0 # matches Disco
    translate_xyz = [
        rotate_dict['Tx'] * TRANSLATION_SCALE, 
        rotate_dict['Ty'] * TRANSLATION_SCALE, 
        -rotate_dict['Tz'] * TRANSLATION_SCALE
    ]
    rotate_xyz = [
        math.radians(rotate_dict['Rx']), 
        math.radians(rotate_dict['Ry']), 
        math.radians(rotate_dict['Rz'])
    ]
    rot_mat = p3d.euler_angles_to_matrix(torch.tensor(rotate_xyz, device=device), "XYZ").unsqueeze(0)
    result, warped_depth = transform_image_3d(prev_img_cv2, depth, rot_mat, translate_xyz, camera_origin, device)
    torch.cuda.empty_cache()
    return result, warped_depth

def anim_warp_depth_3d(depth, rotate_dict, camera_origin, device):
    TRANSLATION_SCALE = 1.0/200.0 # matches Disco
    translate_xyz = [
        rotate_dict['Tx'] * TRANSLATION_SCALE, 
        rotate_dict['Ty'] * TRANSLATION_SCALE, 
        -rotate_dict['Tz'] * TRANSLATION_SCALE
    ]
    rotate_xyz = [
        math.radians(rotate_dict['Rx']), 
        math.radians(rotate_dict['Ry']), 
        math.radians(rotate_dict['Rz'])
    ]
    rot_mat = p3d.euler_angles_to_matrix(torch.tensor(rotate_xyz, device=device), "XYZ").unsqueeze(0)
    warped_depth = transform_depth_3d(depth, rot_mat.to(torch.float32), translate_xyz, camera_origin, device)
    torch.cuda.empty_cache()
    return warped_depth