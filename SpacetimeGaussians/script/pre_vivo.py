# MIT License

# Copyright (c) 2023 OPPO

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import os
from pathlib import Path

import cv2 
import glob 
import tqdm 
import numpy as np 
import shutil
import pickle
import sys 
import argparse


sys.path.append(".")
from thirdparty.gaussian_splatting.utils.my_utils import posetow2c_matrcs, rotmat2qvec
from thirdparty.colmap.pre_colmap import * 
from thirdparty.gaussian_splatting.helper3dg import getcolmapsinglen3d
from script.utils_pre import write_colmap

from script.CondenseDataset import BaseClass



def preparecolmapdynerf(folder, offset=0):
    folderlist = sorted(folder.glob("cam??/"))

    savedir = folder / f"colmap_{offset}" / "input"
    savedir.mkdir(exist_ok=True, parents=True)

    for folder in folderlist:
        imagepath = folder / f"{offset}.jpg"
        imagesavepath = savedir / f"{folder.name}.jpg"

        if (imagesavepath.exists()):
            continue
        else:
            print(f'Does not exist: {imagepath}')

        assert imagepath.exists
        # shutil.copy(imagepath, imagesavepath)
        imagesavepath.symlink_to(imagepath.resolve())


def convertvivotocolmapdb(path, dataset, offset=0, downscale=1):
    """For each image fame (offset) get the pose of each camera

    Note:
        Cam00 to Cam09 correspond to the training cameras in ascending numeric order
        Cam10 to Cam13 is the came for test ddata
    """
    originnumpy = path / "poses_bounds.npy"
    video_paths = sorted(path.glob('cam*.mp4'))


    cameras = []
    id = 0

    for split in [dataset.train_ds, dataset.test_ds]:
        cam_infos = split.cam_infos

        cam_infos = sorted(cam_infos.items(), key=lambda x: int(x[0]))

        for n, c in cam_infos:

            colmapQ = rotmat2qvec(c['R'])
            T = c['T']

            camera = {
                'id': id + 1,
                'filename': f"cam{str(id).zfill(2)}.jpg",
                'w': c['W'],
                'h': c['H'],
                'fx': c['fx'],
                'fy': c['fy'],
                'cx': c['cx'],
                'cy': c['cy'],

                'q': colmapQ,
                't': T,
            }
            id += 1 # update id manually
            cameras.append(camera)
    write_colmap(path, cameras, offset)



if __name__ == "__main__" :
    parser = argparse.ArgumentParser()
 
    parser.add_argument("--videopath", default="", type=str)
    parser.add_argument("--downscale", default=1, type=int)

    args = parser.parse_args()
    videopath = Path(args.videopath)

    startframe = 1
    endframe = 50
    downscale = args.downscale

    print(f"params: startframe={startframe} - endframe={endframe} - downscale={downscale} - videopath={videopath}")


    # # ## step2 prepare colmap input
    print("start preparing colmap image input")
    for offset in range(startframe, endframe):
        preparecolmapdynerf(videopath, offset)

    print("start preparing colmap database input")
    # # ## step 3 prepare colmap db file
    for offset in tqdm.tqdm(range(startframe, endframe), desc="convertdynerftocolmapdb"):
        convertvivotocolmapdb(videopath, BaseClass(datadir="/data/Condense_v2/scenes/A1/"), offset, downscale)

    # ## step 4 run colmap, per frame, if error, reinstall opencv-headless
    for offset in range(startframe, endframe):
        getcolmapsinglen3d(videopath, offset)

