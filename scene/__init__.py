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

import os
import random
import json
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON

class Scene:

    gaussians : GaussianModel

    def __init__(self, args : ModelParams, gaussians : GaussianModel, load_iteration=None, shuffle=True, resolution_scales=[1.0]):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians
        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))


        if os.path.exists(os.path.join(args.source_path, "sparse")): #readColmapSceneInfo
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval, args.llffhold, 
                                                          args.train_num_camera_ratio, args.split_file,
                                                          args.focal_length_scale)
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval, train_num_camera_ratio=args.train_num_camera_ratio)
        else:
            assert False, "Could not recognize scene type!"

        if not self.loaded_iter:
            with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
                dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                if type(scene_info.test_cameras) == dict:
                    for key, value in scene_info.test_cameras.items():
                        camlist.extend(value)
                else:
                    camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            # [Yutong] I comment this. Why do you need to shuffle test cameras?
            #random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        self.cameras_extent = scene_info.nerf_normalization["radius"]

        self.train_cameras = {}
        self.test_cameras = {}
        if isinstance(scene_info.test_cameras, list):
            self.test_cameras = {'test':{}}
        elif isinstance(scene_info.test_cameras, dict):
            self.test_cameras = {k:{} for k in scene_info.test_cameras.keys()}

        for resolution_scale in resolution_scales: #[1.0]
            print('Resolution_scale: ', resolution_scale)
            print("Loading Training Cameras", end=' ')
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args)
            print(f"#={len(self.train_cameras[resolution_scale])}")
            if isinstance(scene_info.test_cameras, list):
                self.test_cameras['test'][resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args)
                print("Loading Test Cameras", f"#={len(self.test_cameras)}")
            elif isinstance(scene_info.test_cameras, dict):
                for key, value in scene_info.test_cameras.items():
                    self.test_cameras[key][resolution_scale] = cameraList_from_camInfos(value, resolution_scale, args)
                    print("Loading Test Cameras", f"Length of {key}: {len(self.test_cameras[key][resolution_scale])}")

        if self.loaded_iter:
            self.gaussians.load_ply(os.path.join(self.model_path,
                                                           "point_cloud",
                                                           "iteration_" + str(self.loaded_iter),
                                                           "point_cloud.ply"))
        else:
            self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)

    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0, test_name="test"):
        return self.test_cameras[test_name][scale]