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
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.oursfull import GaussianModel
from arguments import ModelParams
from helper_train import recordpointshelper, getfisheyemapper
from thirdparty.gaussian_splatting.scene.dataset import FourDGSdataset
from utils.camera_utils import camera_to_JSON, cameraList_from_camInfosv2, cameraList_from_camInfosv2nogt
import random
import torch
from tqdm import tqdm

class Scene:


    gaussians : GaussianModel

    def __init__(self, args : ModelParams, gaussians, load_iteration=None, shuffle=True, resolution_scales=[1.0], multiview=False,duration=50.0, loader="colmap"):
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

        self.train_cameras = {}
        self.test_cameras = {}
        self.video_cameras = {}

        scene_info = sceneLoadTypeCallbacks["Condense"](args.source_path, args.eval)
        dataset_type = "condense"

        self.dataset_type = dataset_type
        self.cameras_extent = scene_info.nerf_normalization["radius"]

        print('Loading Dataset')
        self.train_cameras = FourDGSdataset(scene_info.train_cameras, args, 'condense')
        self.test_cameras = FourDGSdataset(scene_info.test_cameras, args, 'condense')

        print('Datasets full loaded...')

        if self.loaded_iter :
            self.gaussians.load_ply(os.path.join(self.model_path,
                                                           "point_cloud",
                                                           "iteration_" + str(self.loaded_iter),
                                                           "point_cloud.ply"))
        else:
            self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)

        print('Scene finished loading...')

    def get_train_cam_list(self, index):
        # They already do random sampling in the main loop so we dont need to randomly sample w.r.t batch szie
        indexs = self.train_cameras.timedict[index]
        cams = []
        for index in indexs:
            cams.append(self.train_cameras.get_dataset_item(index))
        return cams

    def get_test_cam_list(self):
        camlist = []
        for index in range(len(self.test_cameras.dataset)):
            camlist.append(self.train_cameras.get_dataset_item(index))
        return camlist

    def init_fine(self):
        self.train_cameras.dataset.stage = 'fine'
        self.test_cameras.dataset.stage = 'fine'

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras

    def getTestCameras(self, scale=1.0):
        return self.test_cameras

    # recordpointshelper(model_path, numpoints, iteration, string):
    def recordpoints(self, iteration, string):
        txtpath = os.path.join(self.model_path, "exp_log.txt")
        numpoints = self.gaussians._xyz.shape[0]
        recordpointshelper(self.model_path, numpoints, iteration, string)

    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

 