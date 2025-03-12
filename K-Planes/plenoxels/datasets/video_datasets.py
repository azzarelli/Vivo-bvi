import glob
import json
import logging as log
import math
import os
import time
from collections import defaultdict
from typing import Optional, List, Tuple, Any, Dict

import numpy as np
import torch

from .base_dataset import BaseDataset
from .data_loading import parallel_load_images
from .intrinsics import Intrinsics
from .llff_dataset import load_llff_poses_helper
from .vivo_dataset import CondenseData
from .ray_utils import (
    generate_spherical_poses, create_meshgrid, stack_camera_dirs, get_rays, generate_spiral_path
)
from .synthetic_nerf_dataset import (
    load_360_images, load_360_intrinsics,
)


class Video360Dataset(BaseDataset):
    len_time: int
    max_cameras: Optional[int]
    max_tsteps: Optional[int]
    timestamps: Optional[torch.Tensor]

    def __init__(self,
                 datadir: str,
                 split: str,
                 batch_size: Optional[int] = None,
                 downsample: float = 1.0,
                 keyframes: bool = False,
                 max_cameras: Optional[int] = None,
                 max_tsteps: Optional[int] = None,
                 isg: bool = False,
                 contraction: bool = False,
                 ndc: bool = False,
                 scene_bbox: Optional[List] = None,
                 near_scaling: float = 0.9,
                 ndc_far: float = 2.6):
        self.keyframes = keyframes
        self.max_cameras = max_cameras
        self.max_tsteps = max_tsteps
        self.downsample = downsample
        self.isg = isg
        self.ist = False
        # self.lookup_time = False
        self.per_cam_near_fars = None
        self.global_translation = torch.tensor([0, 0, 0])
        self.global_scale = torch.tensor([1, 1, 1])
        self.near_scaling = near_scaling
        self.ndc_far = ndc_far
        self.median_imgs = None
        if contraction and ndc:
            raise ValueError("Options 'contraction' and 'ndc' are exclusive.")


        if 'scenes' in datadir:
            dset_type = 'vivo'
        elif "lego" in datadir or "dnerf" in datadir:
            dset_type = "synthetic"
        else:
            dset_type = "llff"

        # Note: timestamps are stored normalized between -1, 1.
        if dset_type == 'vivo':
            if split == "render":
                assert ndc, "Unable to generate render poses without ndc: don't know near-far."
                per_cam_poses, per_cam_near_fars, intrinsics, _ = load_llffvideo_poses(
                    datadir, downsample=self.downsample, split='all', near_scaling=self.near_scaling)
                render_poses = generate_spiral_path(
                    per_cam_poses.numpy(), per_cam_near_fars.numpy(), n_frames=300,
                    n_rots=2, zrate=0.5, dt=self.near_scaling, percentile=60)
                self.poses = torch.from_numpy(render_poses).float()
                self.per_cam_near_fars = torch.tensor([[0.4, self.ndc_far]])
                timestamps = torch.linspace(0, 299, len(self.poses))
                imgs = None
            else:
                imgs, poses, timestamps, per_cam_near_fars, intrinsics, self.median_imgs = load_vivo_poses(
                    datadir, downsample=self.downsample, split=split
                )

                imgs = imgs.permute(0, 2, 3,1)
                self.poses = poses.float()
                if contraction:
                    self.per_cam_near_fars = per_cam_near_fars.float()
                else:
                    self.per_cam_near_fars = torch.tensor(
                        [[0.0, self.ndc_far]]).repeat(per_cam_near_fars.shape[0], 1)

            # These values are tuned for the salmon video
            self.global_translation = torch.tensor([0., 0., 0.])
            self.global_scale = torch.tensor([1., 1., 1.])

        imgs = imgs.contiguous()

        self.timestamps = timestamps
        if split == 'train':
            self.timestamps = self.timestamps[:, None, None].repeat(
                1, intrinsics.height, intrinsics.width).reshape(-1)  # [n_frames * h * w]
        assert self.timestamps.min() >= -1.0 and self.timestamps.max() <= 1.0, "timestamps out of range."
        if imgs is not None and imgs.dtype != torch.uint8:
            imgs = (imgs * 255).to(torch.uint8)


        if self.median_imgs is not None and self.median_imgs.dtype != torch.uint8:
            self.median_imgs = (self.median_imgs * 255).to(torch.uint8)


        if split == 'train':
            imgs = imgs.view(-1, imgs.shape[-1])
        elif imgs is not None:
            imgs = imgs.view(-1, intrinsics.height * intrinsics.width, imgs.shape[-1])

        # ISG/IST weights are computed on 4x subsampled data.
        weights_subsampled = int(4 / downsample)
        if scene_bbox is not None:
            scene_bbox = torch.tensor(scene_bbox)
        else:
            scene_bbox = get_bbox(datadir, is_contracted=contraction, dset_type=dset_type)
        super().__init__(
            datadir=datadir,
            split=split,
            batch_size=batch_size,
            is_ndc=ndc,
            is_contracted=contraction,
            scene_bbox=scene_bbox,
            rays_o=None,
            rays_d=None,
            intrinsics=intrinsics,
            imgs=imgs,
            sampling_weights=None,  # Start without importance sampling, by default
            weights_subsampled=weights_subsampled,
        )

        self.isg_weights = None
        self.ist_weights = None
        if split == "train" and dset_type == 'vivo':  # Only use importance sampling with DyNeRF videos
            if os.path.exists(os.path.join(datadir, f"isg_weights.pt")):
                self.isg_weights = torch.load(os.path.join(datadir, f"isg_weights.pt"))
                log.info(f"Reloaded {self.isg_weights.shape[0]} ISG weights from file.")


            else:
                # Precompute ISG weights
                t_s = time.time()
                gamma = 1e-3 if self.keyframes else 2e-2
                self.isg_weights = dynerf_isg_weight(
                    imgs,
                    median_imgs=self.median_imgs, gamma=gamma)
                # Normalize into a probability distribution, to speed up sampling
                self.isg_weights = (self.isg_weights.reshape(-1) / torch.sum(self.isg_weights))
                torch.save(self.isg_weights, os.path.join(datadir, f"isg_weights.pt"))
                t_e = time.time()
                log.info(f"Computed {self.isg_weights.shape[0]} ISG weights in {t_e - t_s:.2f}s.")

            if os.path.exists(os.path.join(datadir, f"ist_weights.pt")):
                self.ist_weights = torch.load(os.path.join(datadir, f"ist_weights.pt"))
                log.info(f"Reloaded {self.ist_weights.shape[0]} IST weights from file.")
            else:
                # Precompute IST weights
                t_s = time.time()
                self.ist_weights = dynerf_ist_weight(
                    imgs,
                    num_cameras=self.median_imgs.shape[0])
                # Normalize into a probability distribution, to speed up sampling
                self.ist_weights = (self.ist_weights.reshape(-1) / torch.sum(self.ist_weights))
                torch.save(self.ist_weights, os.path.join(datadir, f"ist_weights.pt"))
                t_e = time.time()
                log.info(f"Computed {self.ist_weights.shape[0]} IST weights in {t_e - t_s:.2f}s.")

        if self.isg:
            self.enable_isg()

        log.info(f"VideoDataset contracted={self.is_contracted}, ndc={self.is_ndc}. "
                 f"Loaded {self.split} set from {self.datadir}: "
                 f"{len(self.poses)} images of size {self.img_h}x{self.img_w}. "
                 f"Images loaded: {self.imgs is not None}. "
                 f"ISG={self.isg}, IST={self.ist}, weights_subsampled={self.weights_subsampled}. "
                 f"Sampling without replacement={self.use_permutation}. {intrinsics}")

    def enable_isg(self):
        self.isg = True
        self.ist = False
        self.sampling_weights = self.isg_weights
        log.info(f"Enabled ISG weights.")

    def switch_isg2ist(self):
        self.isg = False
        self.ist = True
        self.sampling_weights = self.ist_weights
        log.info(f"Switched from ISG to IST weights.")

    def __getitem__(self, index):
        h = self.intrinsics.height
        w = self.intrinsics.width
        dev = "cpu"
        if self.split == 'train':
            index = self.get_rand_ids(index)  # [batch_size // (weights_subsampled**2)]
            if self.weights_subsampled == 1 or self.sampling_weights is None:
                # Nothing special to do, either weights_subsampled = 1, or not using weights.
                image_id = torch.div(index, h * w, rounding_mode='floor')
                y = torch.remainder(index, h * w).div(w, rounding_mode='floor')
                x = torch.remainder(index, h * w).remainder(w)
            else:
                # We must deal with the fact that ISG/IST weights are computed on a dataset with
                # different 'downsampling' factor. E.g. if the weights were computed on 4x
                # downsampled data and the current dataset is 2x downsampled, `weights_subsampled`
                # will be 4 / 2 = 2.
                # Split each subsampled index into its 16 components in 2D.
                hsub, wsub = h // self.weights_subsampled, w // self.weights_subsampled
                image_id = torch.div(index, hsub * wsub, rounding_mode='floor')
                ysub = torch.remainder(index, hsub * wsub).div(wsub, rounding_mode='floor')
                xsub = torch.remainder(index, hsub * wsub).remainder(wsub)
                # xsub, ysub is the first point in the 4x4 square of finely sampled points
                x, y = [], []
                for ah in range(self.weights_subsampled):
                    for aw in range(self.weights_subsampled):
                        x.append(xsub * self.weights_subsampled + aw)
                        y.append(ysub * self.weights_subsampled + ah)
                x = torch.cat(x)
                y = torch.cat(y)
                image_id = image_id.repeat(self.weights_subsampled ** 2)
                # Inverse of the process to get x, y from index. image_id stays the same.
                index = x + y * w + image_id * h * w
            x, y = x + 0.5, y + 0.5
        else:
            image_id = [index]
            x, y = create_meshgrid(height=h, width=w, dev=dev, add_half=True, flat=True)

        out = {
            "timestamps": self.timestamps[index],      # (num_rays or 1, )
            "imgs": None,
        }
        if self.split == 'train':
            num_frames_per_camera = len(self.imgs) // (len(self.per_cam_near_fars) * h * w)
            camera_id = torch.div(image_id, num_frames_per_camera, rounding_mode='floor')  # (num_rays)
            out['near_fars'] = self.per_cam_near_fars[camera_id, :]
        else:
            out['near_fars'] = self.per_cam_near_fars  # Only one test camera

        if self.imgs is not None:
            out['imgs'] = (self.imgs[index] / 255.0).view(-1, self.imgs.shape[-1])

        c2w = self.poses[image_id]                                    # [num_rays or 1, 3, 4]
        camera_dirs = stack_camera_dirs(x, y, self.intrinsics, opengl_camera=False)  # [num_rays, 3]
        out['rays_o'], out['rays_d'] = get_rays(
            camera_dirs, c2w, ndc=self.is_ndc, ndc_near=1.0, intrinsics=self.intrinsics,
            normalize_rd=True)  # [num_rays, 3]

        imgs = out['imgs']
        # Decide BG color
        bg_color = torch.ones((1, 3), dtype=torch.float32, device=dev)
        if self.split == 'train' and imgs.shape[-1] == 4:
            bg_color = torch.rand((1, 3), dtype=torch.float32, device=dev)
        out['bg_color'] = bg_color
        # Alpha compositing
        if imgs is not None and imgs.shape[-1] == 4:
            imgs = imgs[:, :3] * imgs[:, 3:] + bg_color * (1.0 - imgs[:, 3:])
        out['imgs'] = imgs

        return out

def get_bbox(datadir: str, dset_type: str, is_contracted=False) -> torch.Tensor:
    """Returns a default bounding box based on the dataset type, and contraction state.

    Args:
        datadir (str): Directory where data is stored
        dset_type (str): A string defining dataset type (e.g. synthetic, llff)
        is_contracted (bool): Whether the dataset will use contraction

    Returns:
        Tensor: 3x2 bounding box tensor
    """
    if is_contracted:
        radius = 2
    elif dset_type == 'synthetic':
        radius = 1.5
    elif dset_type == 'llff':
        return torch.tensor([[-3.0, -1.67, -1.2], [3.0, 1.67, 1.2]])
    else:
        radius = 1.3
    return torch.tensor([[-radius, -radius, -radius], [radius, radius, radius]])


def fetch_360vid_info(frame: Dict[str, Any]):
    timestamp = None
    fp = frame['file_path']
    if '_r' in fp:
        timestamp = int(fp.split('t')[-1].split('_')[0])
    if 'r_' in fp:
        pose_id = int(fp.split('r_')[-1])
    else:
        pose_id = int(fp.split('r')[-1])
    if timestamp is None:  # will be None for dnerf
        timestamp = frame['time']
    return timestamp, pose_id


def load_vivo_poses(datadir: str,
                         downsample: float,
                         split: str,) -> Tuple[
                            torch.Tensor, torch.Tensor, Intrinsics, List[str]]:
    """
    We need the following:
        Poses, intrinsics, near-fars, median image and image paths

    I'd like to use ISG and IST so we need the median images
    Though maybe we can load the images on the fly

    """
    print(f'Loading condense with {downsample} x downsample')
    data = CondenseData(datadir, split, downsample=downsample)

    # Run this to visualize the poses
    # data.viz_poses_for_KPlanes()
    # exit()

    median_images = data.load_median_images() if split == 'train' else None

    intrinsics = data.get_KPlanes_Intrinsics()
    intrinsics.scale(1./downsample)

    # Poses M, 4, 4 - in c2w form
    imgs, poses, times = data.get_KPlanes_poses()

    # Per Camera near and far ray bounds
    if split == 'train':
        N = 10
    else:
        N = 4
    near_fars = torch.tensor([[0.5, 8.0] for i in range(N)])
    # Spare function for centering poses
    # poses, pose_avg = center_poses(poses)

    return imgs, poses, times, near_fars, intrinsics ,median_images



@torch.no_grad()
def dynerf_isg_weight(imgs, median_imgs, gamma):
    # imgs is [num_cameras * num_frames, h, w, 3]
    # median_imgs is [num_cameras, h, w, 3]
    assert imgs.dtype == torch.uint8
    assert median_imgs.dtype == torch.uint8
    num_cameras, h, w, c = median_imgs.shape
    squarediff = (
        imgs.view(num_cameras, -1, h, w, c)
            .float()  # creates new tensor, so later operations can be in-place
            .div_(255.0)
            .sub_(
                median_imgs[:, None, ...].float().div_(255.0)
            )
            .square_()  # noqa
    )  # [num_cameras, num_frames, h, w, 3]
    # differences = median_imgs[:, None, ...] - imgs.view(num_cameras, -1, h, w, c)  # [num_cameras, num_frames, h, w, 3]
    # squarediff = torch.square_(differences)
    psidiff = squarediff.div_(squarediff + gamma**2)
    psidiff = (1./3) * torch.sum(psidiff, dim=-1)  # [num_cameras, num_frames, h, w]
    return psidiff  # valid probabilities, each in [0, 1]


@torch.no_grad()
def dynerf_ist_weight_old(imgs, num_cameras, alpha=0.1, frame_shift=25):  # DyNerf uses alpha=0.1
    print('Determining it weight')
    assert imgs.dtype == torch.uint8
    N, h, w, c = imgs.shape
    frames = imgs.view(num_cameras, -1, h, w, c).float()  # [num_cameras, num_timesteps, h, w, 3]
    max_diff = None
    shifts = list(range(frame_shift + 1))[1:]
    for shift in shifts:
        shift_left = torch.cat([frames[:, shift:, ...], torch.zeros(num_cameras, shift, h, w, c).half()], dim=1)
        shift_right = torch.cat([torch.zeros(num_cameras, shift, h, w, c).half(), frames[:, :-shift, ...]], dim=1)
        mymax = torch.maximum(torch.abs_(shift_left - frames), torch.abs_(shift_right - frames))
        if max_diff is None:
            max_diff = mymax
        else:
            max_diff = torch.maximum(max_diff, mymax)  # [num_timesteps, h, w, 3]
    max_diff = torch.mean(max_diff, dim=-1)  # [num_timesteps, h, w]
    max_diff = max_diff.clamp_(min=alpha)
    return max_diff

import gc
@torch.no_grad()
def dynerf_ist_weight(imgs, num_cameras, alpha=0.1, frame_shift=25):
    print('Determining it weight')
    assert imgs.dtype == torch.uint8

    N, h, w, c = imgs.shape
    frames = imgs.view(num_cameras, -1, h, w, c).to(torch.float16)  # Use float16 to reduce memory

    max_diff = torch.zeros(frames.shape[1], h, w, device=imgs.device, dtype=torch.float16)  # Reduce dtype size

    for shift in range(1, frame_shift + 1):
        shift_left = torch.zeros_like(frames)
        shift_right = torch.zeros_like(frames)

        shift_left[:, :-shift, ...] = frames[:, shift:, ...]
        shift_right[:, shift:, ...] = frames[:, :-shift, ...]

        with torch.no_grad():  # Disable gradient tracking for lower memory usage
            mymax = torch.maximum(torch.abs(shift_left - frames), torch.abs(shift_right - frames))
            max_diff = torch.maximum(max_diff, torch.mean(mymax, dim=-1))

        # Clean up unused tensors
        del shift_left, shift_right, mymax
        torch.cuda.empty_cache()  # Free GPU memory
        gc.collect()  #
    return max_diff.clamp_(min=alpha)  # [num_timesteps, h, w]