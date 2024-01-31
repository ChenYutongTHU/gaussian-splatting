#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from scene.cameras import Camera
import numpy as np
from utils.general_utils import PILtoTorch
from utils.graphics_utils import fov2focal
from tqdm import tqdm 
import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal

WARNED = False
class CameraDataset(Dataset):
    def __init__(self, cam_infos, resolution_scale, args, is_training):
        self.cam_infos = cam_infos
        self.resolution_scale = resolution_scale
        self.args = args

        if args.rnd_background and is_training:
            self.bg_color = 'rnd'
        elif args.white_background:
            self.bg_color = np.array([1,1,1]) 
        else:
            self.bg_color = np.array([0, 0, 0])

    def __len__(self):
        return len(self.cam_infos)
    
    def __getitem__(self, idx):
        image = Image.open(self.cam_infos[idx].image_path)
        im_data = np.array(image.convert("RGBA"))
        #bg = np.array([1,1,1]) if self.cam_infos[idx].white_background else np.array([0, 0, 0])
        if type(self.bg_color)==str and self.bg_color == 'rnd':
            bg = np.random.rand(3)
        else:
            bg = self.bg_color
        norm_data = im_data / 255.0
        arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
        image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")
        return loadCam(self.args, idx, self.cam_infos[idx], self.resolution_scale, 
                       image=image, data_device='cpu', bg=bg)



def loadCam(args, id, cam_info, resolution_scale, bg, image=None, data_device=None):
    if image is None:
        image = cam_info.image 
        FovY = cam_info.FovY
    else:
        image = image 
        width, height = image.size
        FovY = focal2fov(fov2focal(cam_info.FovX, width), height)
    
    data_device = args.data_device if data_device is None else data_device

    orig_w, orig_h = image.size

    if args.resolution in [1, 2, 4, 8]:
        resolution = round(orig_w/(resolution_scale * args.resolution)), round(orig_h/(resolution_scale * args.resolution))
    else:  # should be a type that converts to float
        if args.resolution == -1:
            if orig_w > 1600:
                global WARNED
                if not WARNED:
                    print("[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
                        "If this is not desired, please explicitly specify '--resolution/-r' as 1")
                    WARNED = True
                global_down = orig_w / 1600
            else:
                global_down = 1
        else:
            global_down = orig_w / args.resolution

        scale = float(global_down) * float(resolution_scale)
        resolution = (int(orig_w / scale), int(orig_h / scale))

    resized_image_rgb = PILtoTorch(image, resolution)

    gt_image = resized_image_rgb[:3, ...]
    loaded_mask = None

    if resized_image_rgb.shape[1] == 4:
        loaded_mask = resized_image_rgb[3:4, ...]
    return Camera(colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T, 
                  FoVx=cam_info.FovX, FoVy=FovY, 
                  image=gt_image, gt_alpha_mask=loaded_mask,
                  image_name=cam_info.image_name, uid=id, data_device=data_device, bg=bg)


def Camera_Collate_fn(batch):
    return batch[0]

def cameraList_from_camInfos(cam_infos, resolution_scale, args, shuffle=False):
    if args.dataset_type.lower() == 'list': #preload image
        camera_list = []
        for id, c in tqdm(enumerate(cam_infos)):
            camera_list.append(loadCam(args, id, c, resolution_scale))
        return camera_list
    elif args.dataset_type.lower() == 'loader':
        dataset = CameraDataset(cam_infos, resolution_scale, args, is_training=shuffle)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=shuffle, collate_fn=Camera_Collate_fn, num_workers=4)
        return dataloader



def camera_to_JSON(id, camera : Camera):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = camera.R.transpose()
    Rt[:3, 3] = camera.T
    Rt[3, 3] = 1.0

    W2C = np.linalg.inv(Rt)
    pos = W2C[:3, 3]
    rot = W2C[:3, :3]
    serializable_array_2d = [x.tolist() for x in rot]
    camera_entry = {
        'id' : id,
        'img_name' : camera.image_name,
        'width' : camera.width,
        'height' : camera.height,
        'position': pos.tolist(),
        'rotation': serializable_array_2d,
        'fy' : fov2focal(camera.FovY, camera.height) if camera.height is not None else None,
        'fx' : fov2focal(camera.FovX, camera.width) if camera.width is not None else None,
        'FovX' : camera.FovX,
        'FovY' : camera.FovY,
    }
    return camera_entry
