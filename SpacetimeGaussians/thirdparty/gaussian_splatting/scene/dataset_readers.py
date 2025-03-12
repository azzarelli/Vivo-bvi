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
import sys
from PIL import Image
from typing import NamedTuple
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import json
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
from utils.graphics_utils import BasicPointCloud
import glob
import natsort
from simple_knn._C import distCUDA2
import torch

class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int
    near: float
    far: float
    timestamp: float
    pose: np.array 
    hpdirecitons: np.array
    cxr: float
    cyr: float

class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str
    
def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}


def normalize(v):
    return v / np.linalg.norm(v)


def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    times = np.vstack([vertices['t']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals, times=times)

def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
             ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
             ('red', 'f4'), ('green', 'f4'), ('blue', 'f4')]

    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    # breakpoint()
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)


from tqdm import tqdm
def format_condense_infos(dataset,split):
    # loading
    cameras = []

    tempData = dataset.cam_infos[list(dataset.cam_infos.keys())[0]]

    print(['W'])
    for idx, item in tqdm(enumerate(dataset)):
        path, (R,T), time = item
        ccx, ccy = dataset.load_cc(idx)
        FovX, FovY = dataset.load_fov(idx)

        cameras.append(
            CameraInfo(
                uid=idx,
                R=R, T=T,
                FovY=FovY, FovX=FovX,
                image_path=path,
                width=tempData['W'], height=tempData['H'],
                timestamp = time,
                near=0.05, far=20.,
                hpdirecitons=1,
                pose=1,
                cxr=ccx, cyr=ccy,
                image=None, image_name=dataset.num_frames
            )
        )

    return cameras


def readCondenseSceneInfo(datadir, eval):
    from thirdparty.gaussian_splatting.scene.condense_dataset import CondenseData
    import open3d as o3d

    train_cam_infos = CondenseData(datadir, split='train')

    test_cam_infos = CondenseData(datadir, split='test')

    train_cam_infos_ = format_condense_infos(train_cam_infos, "train")
    test_cam_infos_ = format_condense_infos(test_cam_infos, "train")

    nerf_normalization = getNerfppNorm(train_cam_infos_)

    pcd = o3d.io.read_point_cloud(os.path.join(datadir, f'pcds/sparse/000000.ply'))
    xyz = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)
    # colors = np.random.random((xyz.shape[0], 3))

    pcd = BasicPointCloud(points=xyz, colors=colors, normals=np.zeros((xyz.shape[0], 3)), times=np.zeros((xyz.shape[0], 1)))

    ply_path = os.path.join(datadir, "pcds/fused.ply")
    storePly(ply_path, xyz, colors * 255)

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos_,
                           test_cameras=test_cam_infos_,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path,
                           )
    print(f"Loaded condense dataset...")

    return scene_info

sceneLoadTypeCallbacks = {
    "Condense": readCondenseSceneInfo,
}


