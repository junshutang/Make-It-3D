"""Compute depth maps for images in the input folder.
"""
import os
import glob
import sys
import torch
import cv2
import argparse
import math
import numpy as np
from tqdm import tqdm
sys.path.append('DPT/')
from torchvision.transforms import Compose

from util.io import read_image
from dpt.models import DPTDepthModel
from dpt.midas_net import MidasNet_large
from dpt.transforms import Resize, NormalizeImage, PrepareForNet

from warp_utils import anim_warp_3d, anim_warp_depth_3d
from torchvision.utils import make_grid
#from util.misc import visualize_attention

def tensor_to_img(img, **kwargs):
    grid = make_grid(img, **kwargs)
    ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    return ndarr[:, :, ::-1]

def tensor_to_depth(img, **kwargs):
    grid = make_grid(img, **kwargs)
    ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    return ndarr[:, :, ::-1]

def run(input_path, output_path, model_path, model_type="dpt_hybrid", optimize=True):
    """Run MonoDepthNN to compute depth maps.

    Args:
        input_path (str): path to input folder
        output_path (str): path to output folder
        model_path (str): path to saved model
    """
    print("initialize")

    # select device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device: %s" % device)

    # load network
    if model_type == "dpt_large":  # DPT-Large
        net_w = net_h = 384
        model = DPTDepthModel(
            path=model_path,
            backbone="vitl16_384",
            non_negative=True,
            enable_attention_hooks=False,
        )
        normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    elif model_type == "dpt_hybrid":  # DPT-Hybrid
        net_w = net_h = 384
        model = DPTDepthModel(
            path=model_path,
            backbone="vitb_rn50_384",
            non_negative=True,
            enable_attention_hooks=False,
        )
        normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    elif model_type == "dpt_hybrid_kitti":
        net_w = 1216
        net_h = 352

        model = DPTDepthModel(
            path=model_path,
            scale=0.00006016,
            shift=0.00579,
            invert=True,
            backbone="vitb_rn50_384",
            non_negative=True,
            enable_attention_hooks=False,
        )

        normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    elif model_type == "dpt_hybrid_nyu":
        net_w = 640
        net_h = 480

        model = DPTDepthModel(
            path=model_path,
            scale=0.000305,
            shift=0.1378,
            invert=True,
            backbone="vitb_rn50_384",
            non_negative=True,
            enable_attention_hooks=False,
        )

        normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    elif model_type == "midas_v21":  # Convolutional model
        net_w = net_h = 384

        model = MidasNet_large(model_path, non_negative=True)
        normalization = NormalizeImage(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
    else:
        assert (
            False
        ), f"model_type '{model_type}' not implemented, use: --model_type [dpt_large|dpt_hybrid|dpt_hybrid_kitti|dpt_hybrid_nyu|midas_v21]"

    transform = Compose(
        [
            Resize(
                net_w,
                net_h,
                resize_target=None,
                keep_aspect_ratio=True,
                ensure_multiple_of=32,
                resize_method="minimal",
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            normalization,
            PrepareForNet(),
        ]
    )
    
    ori_transform = Compose(
        [
            Resize(
                512,
                512,
                resize_target=None,
                keep_aspect_ratio=True,
                ensure_multiple_of=32,
                resize_method="minimal",
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            normalization,
            PrepareForNet(),
        ]
    )
    trajectory = []
    k=0
    for t in np.linspace(0, 10, 36):
        tx = 0
        ty = 0
        tz = 0
        rx = 10 * math.cos(2*math.pi*t/10)
        ry = 10 * math.sin(2*math.pi*t/10)
        rz = 0
        trajectory.append((k, tx, ty, tz, rx, ry, rz))
        k+=1

    model.eval()

    if optimize == True and device == torch.device("cuda"):
        model = model.to(memory_format=torch.channels_last)
        model = model.half()

    model.to(device)

    # get input
    # img_names = glob.glob(os.path.join(input_path, "*"))
    # num_images = len(img_names)
    img_name = input_path
    # create output folder
    os.makedirs(output_path, exist_ok=True)

    # print("start processing")
    # for ind, img_name in enumerate(img_names):
    #     if os.path.isdir(img_name):
    #         continue

    #     print("  processing {} ({}/{})".format(img_name, ind + 1, num_images))
    #     # input

    img = read_image(img_name)

    img_input = transform({"image": img})["image"]
    img_tensor = ori_transform({"image": img})["image"]
    # compute
    with torch.no_grad():
        sample = torch.from_numpy(img_input).to(device).unsqueeze(0)
        img_tensor = torch.from_numpy(img_tensor).to(device).unsqueeze(0)
        
        if optimize == True and device == torch.device("cuda"):
            sample = sample.to(memory_format=torch.channels_last)
            sample = sample.half()

        prediction = model.forward(sample) # [1, 384, 384]
        
        prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img.shape[:2],
                mode="bicubic",
                align_corners=False,
            ) # [1, 1, 512, 512]
        # prediction = (prediction - prediction.min()) + 1
        
        print(prediction.min(), prediction.max())
        print(prediction)
        prediction = (prediction - prediction.min()) / (prediction.max() - prediction.min()) + 0.5
        camera_origin = torch.tensor([0., 0., 0]).to(device)
        rotate_dict = {
            'Tx': 0,
            'Ty': 0,
            'Tz': 0,
            'Rx': 0,
            'Ry': 0,
            'Rz': 0,
        }
        image_frames = []
        depth_frames = []
        fine_name = img_name.replace('png', 'mp4')
        fine_depth_name = img_name.replace('.png', '_depth.mp4')
        for k, tx, ty, tz, rx, ry, rz in tqdm(trajectory):
            rotate_dict['Tx'] = tx
            rotate_dict['Ty'] = ty
            rotate_dict['Tz'] = tz
            rotate_dict['Rx'] = rx
            rotate_dict['Ry'] = ry
            rotate_dict['Rz'] = rz
            # continue
            out_image_tensor, out_depth_tensor = anim_warp_3d(img_tensor, prediction, rotate_dict, camera_origin, device)
            out_image = tensor_to_img(out_image_tensor, normalize=True)
            out_depth = tensor_to_img(out_depth_tensor, normalize=True)
            image_frames.append(out_image)
            depth_frames.append(out_depth)
        # cv2.imwrite('warp_output.png', out_image)
        height, width, layers = image_frames[0].shape
        output_video = cv2.VideoWriter(fine_name, cv2.VideoWriter_fourcc("m", "p", "4", "v"), 24, (width, height))
        output_depth_video = cv2.VideoWriter(fine_depth_name, cv2.VideoWriter_fourcc("m", "p", "4", "v"), 24, (width, height))
        for frame in image_frames:
            output_video.write(np.array(frame))
            
        for depth_frame in depth_frames:
            output_depth_video.write(np.array(depth_frame))

        output_video.release()
        output_depth_video.release()


    print("finished")

def warp_depth_fun(input_depth, theta, phi, device):

    theta_180 = theta / math.pi * 180.0 - 90.0
    phi_180 = phi / math.pi * 180.0 - 20.0

    # get input
    prediction = input_depth
    # prediction = (prediction - prediction.min()) + 0.5
    camera_origin = torch.tensor([0., 0., 0]).to(device)
    rotate_dict = {
        'Tx': 0,
        'Ty': 0,
        'Tz': 0,
        'Rx': theta_180,
        'Ry': -phi_180,
        'Rz': 0,
    }
    out_depth_tensor = anim_warp_depth_3d(prediction, rotate_dict, camera_origin, device)
    
    return out_depth_tensor

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-i", "--input_path", default="test_img/house.png", help="folder with input images"
    )

    parser.add_argument(
        "-o",
        "--output_path",
        default="output_monodepth",
        help="folder for output images",
    )

    parser.add_argument(
        "-m", "--model_weights", default=None, help="path to model weights"
    )

    parser.add_argument(
        "-t",
        "--model_type",
        default="dpt_hybrid",
        help="model type [dpt_large|dpt_hybrid|midas_v21]",
    )

    parser.add_argument("--kitti_crop", dest="kitti_crop", action="store_true")
    parser.add_argument("--absolute_depth", dest="absolute_depth", action="store_true")

    parser.add_argument("--optimize", dest="optimize", action="store_true")
    parser.add_argument("--no-optimize", dest="optimize", action="store_false")

    parser.set_defaults(optimize=True)
    parser.set_defaults(kitti_crop=False)
    parser.set_defaults(absolute_depth=False)

    args = parser.parse_args()

    default_models = {
        "midas_v21": "DPT/weights/midas_v21-f6b98070.pt",
        "dpt_large": "DPT/weights/dpt_large-midas-2f21e586.pt",
        "dpt_hybrid": "DPT/weights/dpt_hybrid-midas-501f0c75.pt",
        "dpt_hybrid_kitti": "DPT/weights/dpt_hybrid_kitti-cb926ef4.pt",
        "dpt_hybrid_nyu": "DPT/weights/dpt_hybrid_nyu-2ce69ec7.pt",
    }

    if args.model_weights is None:
        args.model_weights = default_models[args.model_type]

    # set torch options
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    # compute depth maps
    run(
        args.input_path,
        args.output_path,
        args.model_weights,
        args.model_type,
        args.optimize,
    )
