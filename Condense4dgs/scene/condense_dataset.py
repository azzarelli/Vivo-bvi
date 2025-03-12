import os
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from utils.graphics_utils import focal2fov
from torchvision import transforms as T
import torch.nn.functional as F
import json
import torch

import cv2

def to_tensor(x, shape=None, device="cpu"):
    # convert the input to torch.tensor
    if shape != None:
        return torch.tensor(x, dtype=torch.float32).view(shape).to(device)
    else:
        return torch.tensor(x, dtype=torch.float32).to(device)


def default_deform_tracking(config, device):
    """ to apply the inverse of [R|T] to the mesh """
    R = to_tensor(config['orientation'], (3, 3), device)  # transpose is omitted to make it column-major
    invT = R @ -to_tensor(config['origin'], (3, 1), device)
    space = to_tensor(config['spacing'], (3,), device)
    dimen = to_tensor(config['dimensions'], (3,), device)

    # offset initialized to zeros
    offset = torch.zeros(invT.size()).to(device)
    offset[1] -= space[1] * (dimen[1] / 2.0)
    offset[2] -= space[2] * (dimen[2] / 2.0)

    T = invT + offset
    return R.unsqueeze(0), T.unsqueeze(0)



def generate_undistortion_grid(K, coeffs, w, h, device='cpu'):
    """
    Generate a grid for undistorting an image with 8 distortion coefficients.

    Parameters:
        K (torch.Tensor): Camera matrix (3x3).
        coeffs (torch.Tensor): Distortion coefficients (k1, k2, p1, p2, k3, k4, k5, k6).
        w (int): Image width.
        h (int): Image height.
        device (str): Device for the tensors ('cpu' or 'cuda').

    Returns:
        torch.Tensor: Undistortion grid of shape (1, h, w, 2).
    """
    # Generate pixel grid
    i = torch.linspace(0, h - 1, h, device=device)
    j = torch.linspace(0, w - 1, w, device=device)
    jj, ii = torch.meshgrid(j, i, indexing='xy')

    # Normalize coordinates to [-1, 1]
    x = (jj - K[0, 2]) / K[0, 0]
    y = (ii - K[1, 2]) / K[1, 1]

    # Calculate radial distance squared
    r2 = x ** 2 + y ** 2
    r4 = r2 ** 2
    r6 = r2 ** 3

    # Radial distortion
    radial = 1 + coeffs[0] * r2 + coeffs[1] * r4 + coeffs[4] * r6 + coeffs[5] * r4 + coeffs[6] * r6

    # Tangential distortion
    x_tangential = 2 * coeffs[2] * x * y + coeffs[3] * (r2 + 2 * x ** 2)
    y_tangential = 2 * coeffs[3] * x * y + coeffs[2] * (r2 + 2 * y ** 2)

    # Distorted coordinates
    x_distorted = x * radial + x_tangential
    y_distorted = y * radial + y_tangential

    # Convert to sampling grid format
    x_mapped = (x_distorted * K[0, 0] + K[0, 2]) / (w - 1) * 2 - 1
    y_mapped = (y_distorted * K[1, 1] + K[1, 2]) / (h - 1) * 2 - 1

    grid = torch.stack((x_mapped, y_mapped), dim=-1).unsqueeze(0)  # Shape (1, h, w, 2)
    return grid

def decompose_dataset(datadir, rotation_correction, split='test', visualise_poses=False):
    with open(os.path.join(datadir, "calibration.json")) as f:
        calib = json.load(f)

    # Get the camera names for the current folder
    cam_names = os.listdir(os.path.join(datadir, f"{split}/"))

    with open(os.path.join(datadir, "capture-area.json")) as f:
        cap_area_config = json.load(f)


    poses = {}
    for ii, c in enumerate(cam_names):
        if ii == 0:
            # We also may need the distortion parameters which we can get from the per-frame meta files
            metadepth_fp = os.path.join(datadir, f'{split}/{c}/meta/000000.depth.json')
            metacol_fp = os.path.join(datadir, f'{split}/{c}/meta/000000.color.json')

            with open(metacol_fp, 'r') as f:
                color_distortion = json.load(f)['imageMetadata']['intrinsics']['distortion']

        meta = calib['cameras'][c]

        depth_ex = meta['depth_extrinsics']
        col2depth_ex = meta['colour_to_depth_extrinsics']

        # Construct w2c transform for depth images
        M_depth = torch.eye(4)
        M_depth[:3, :3] = torch.tensor(depth_ex['orientation']).view((3, 3)).mT
        M_depth[:3, 3] = torch.tensor(depth_ex['translation']).view((3, 1))[:, 0]

        # Construct w2c transform for depth images
        M_col = torch.eye(4)
        M_col[:3, :3] = torch.tensor(col2depth_ex['orientation']).view((3, 3)).mT
        M_col[:3, 3] = torch.tensor(col2depth_ex['translation']).view((3, 1))[:, 0]

        R_m, T_m = default_deform_tracking(cap_area_config, 'cpu')
        M_m = torch.eye(4)
        M_m[:3, :3] = torch.tensor(R_m).view((3, 3))
        M_m[:3, 3] = torch.tensor(T_m).view((3, 1))[:, 0]


        # Generate color (c2w transform) extrinsics for
        M = M_col.inverse() @ M_depth.inverse() @ M_m.inverse()
        M = M #.inverse()
        T = M[:3, 3].numpy()
        R = M[:3, :3].numpy()
        R = R.T

        M_d = M_depth.inverse() @ M_m.inverse()
        M_d = M_d #.inverse()
        T_d = M_d[:3, 3].numpy()
        R_d = M_d[:3, :3].numpy()
        R_d = R_d.T

        H = meta['colour_intrinsics']['height']
        W = meta['colour_intrinsics']['width']
        focal = [meta['colour_intrinsics']['fx'], meta['colour_intrinsics']['fy']]
        focal_depth = [meta['depth_intrinsics']['fx'], meta['depth_intrinsics']['fy']]

        K = np.array([[focal[0], 0, meta['colour_intrinsics']['ppx']], [0, focal[1], meta['colour_intrinsics']['ppy']], [0, 0, 1]])
        grid = generate_undistortion_grid(K, color_distortion, W, H, device='cpu')

        poses[c] = {
            'H': H, 'W': W,
            'focal': focal,
            'FovX': focal2fov(focal[0], W),
            'FovY': focal2fov(focal[1], H),
            'R': R,
            'T': T,
            'cx': meta['colour_intrinsics']['ppx'],
            'cy': meta['colour_intrinsics']['ppy'],
            'grid': grid,
        }


    return poses

import matplotlib.pyplot as plt
from torchvision.transforms import ToPILImage

class DilationTransform:
    def __init__(self, kernel_size=(10, 10)):
        # Create a structuring element (kernel) for dilation
        self.kernel = torch.ones(*kernel_size).unsqueeze(0).unsqueeze(0).cuda()  # 1x1xHxW kernel

    def __call__(self, img):
        img = img.unsqueeze(0).unsqueeze(0)  # [H, W] -> [1, 1, H, W]
        dilated_img = F.conv2d(img, self.kernel, padding='same').squeeze(0).squeeze(0)

        dilated_img[dilated_img > 0] = 255.

        # To display the function
        # self.display(img, dilated_img)

        return dilated_img

    def display(self, original_img, dilated_img):
        original_img = ToPILImage()(original_img.squeeze(0).squeeze(0))
        dilated_img = ToPILImage()(dilated_img)

        plt.figure(figsize=(10, 5))

        # Original Image
        plt.subplot(1, 2, 1)
        plt.title("Original Image")
        plt.imshow(original_img)
        plt.axis('off')

        # Dilated Image
        plt.subplot(1, 2, 2)
        plt.title("Dilated Image")
        plt.imshow(dilated_img)
        plt.axis('off')

        plt.tight_layout()
        plt.show()
        exit()



class CondenseData(Dataset):
    def __init__(
            self,
            datadir,
            split='train',
            downsample=1.0
    ):

        if split == 'train':
            self.image_type_folder = "color_corrected"
        elif split == 'test':
            self.image_type_folder = "masks"
        self.img_wh = (
            int(2560 / downsample),
            int(1440 / downsample),
        )

        # 4x upsample
        self.depth_wh = (
            int(640 /downsample),
            int(576 /downsample),
        )

        self.root_dir = datadir
        self.split = split
        self.num_frames = 0
        with open(os.path.join(datadir, f"rotation_correction.json")) as f:
            self.rotation_correction = json.load(f)

        self.cam_infos = decompose_dataset(datadir, self.rotation_correction, split=split ) #, visualise_poses=True)
        self.grids = []

        self.transform = T.ToTensor()


        self.image_paths, self.image_poses, self.image_times, self.fovs = self.load_images_path(self.root_dir, self.split)
        self.pcd_paths = self.load_pcd_path()



        # Finally figure out idx of coarse image
        self.stage = 'coarse'
        target = '000000'
        self.coarse_idxs = [id for id, f in enumerate(self.image_paths) if target in f]


    def load_images_path(self, cam_folder, split,  stage='fine'):
        image_paths = []
        image_poses = []
        image_times = []
        FOVs = []
        for cam_info in self.cam_infos:
            meta = self.cam_infos[cam_info]


            R = meta['R']
            T = meta['T']

            fovx = meta['FovX']
            fovy = meta['FovY']

            fp = os.path.join(cam_folder, f"{split}/{cam_info}/{self.image_type_folder}")

            time_max = len(os.listdir(fp))
            cnt = 0
            for img_fp in os.listdir(fp):
                img_fp_ = os.path.join(fp, img_fp)
                image_paths.append(img_fp_)
                image_poses.append((R, T))
                image_times.append(float(int(img_fp.split('.')[0]) / time_max))
                FOVs.append((fovx, fovy))
                self.grids.append(cam_info)
                cnt+= 1

        self.num_frames = cnt
        return image_paths, image_poses, image_times, FOVs

    def load_pcd_path(self):
        fp = os.path.join(self.root_dir, f"pcds/sparse/")

        fps = []
        for f in os.listdir(fp):
            fps.append(os.path.join(fp, f))

        return fps

    def unditort_pred(self, img, index):
        return F.grid_sample(img.unsqueeze(0),self.cam_infos[self.grids[index]]['grid'].cuda(), mode='nearest', align_corners=True)


    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        if self.stage == 'fine':
            path = self.image_paths[index]
            pose = self.image_poses[index]
            time = self.image_times[index]
        elif self.stage == 'coarse':
            idx = index % len(self.coarse_idxs)
            path = self.image_paths[self.coarse_idxs[idx]]
            assert '000000' in path, 'Not an initial frame'
            pose = self.image_poses[self.coarse_idxs[idx]]
            time = self.image_times[self.coarse_idxs[idx]]

        img = Image.open(path)
        # img = img.resize(self.img_wh, Image.LANCZOS)
        img = self.transform(img)
        return img, pose, time

    def get_pcd_path(self, index):
        try:
            return self.pcd_paths[index % self.num_frames]
        except:
            return ''

    def load_pose(self, index):
        return self.image_poses[index]

    def load_fov(self, index):
        return self.fovs[index]
