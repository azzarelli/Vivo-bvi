from .intrinsics import Intrinsics

import os
from torch.utils.data import Dataset
from torchvision import transforms as T
import json
import torch
import cv2
import numpy as np
import shutil
from tqdm import tqdm
from PIL import Image
import torch.nn.functional as F

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

def decompose_dataset(datadir, split='test'):
    # Open calibration file
    with open(os.path.join(datadir, "calibration.json")) as f:
        calib = json.load(f)

    # Get the camera names for the current folder
    cam_names = os.listdir(os.path.join(datadir, f"{split}/"))

    # Open capture area folder
    with open(os.path.join(datadir, "capture-area.json")) as f:
        cap_area_config = json.load(f)

    poses = {}
    for ii, c in enumerate(cam_names):

        if ii == 0:
            # We also may need the distortion parameters which we can get from the per-frame meta files
            metadepth_fp = os.path.join(datadir, f'{split}/{c}/meta/000000.depth.json')
            metacol_fp = os.path.join(datadir, f'{split}/{c}/meta/000000.color.json')
            with open(metadepth_fp, 'r') as f:
                depth_distortion = json.load(f)['imageMetadata']['intrinsics']['distortion']

            with open(metacol_fp, 'r') as f:
                color_distortion = json.load(f)['imageMetadata']['intrinsics']['distortion']

        meta = calib['cameras'][c]

        depth_ex = meta['depth_extrinsics']
        col2depth_ex = meta['colour_to_depth_extrinsics']

        """Depth cameras are maped w.r.t the capture area (red markings), so their positional and rotational components are projected
        in relation to the capture area's position and rotation.
        """
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

        # Note that these parameters are in World space (I think...lol at least im not doing 4-D projective geometry (mdr))
        #   so they correspond to c2w transform in the form of (r.T @ (x-t).T).T
        #   and the w2c in the form of (r @ x.T).T + t
        poses[c] = {
            'H': meta['colour_intrinsics']['height'], 'W': meta['colour_intrinsics']['width'],
            'fx': meta['colour_intrinsics']['fx'],
            'fy': meta['colour_intrinsics']['fy'],
            'M': M,
            'cx': meta['colour_intrinsics']['ppx'],
            'cy': meta['colour_intrinsics']['ppy'],
            'distortion': color_distortion,
        }

    """Final note (I promise):
        The preprocessing is NOT complete until we rotate the images correctly (yes they need to be rotated). 
        I haven't baked this into the transforms due to my own stupidity
    """
    return poses


def rotation_distortion_correction(fp, rc, K, coeffs, w, h):
    """Rotate and undistort the input image

        Arguments:
            fp: string, file path
            rc: int, -1, 1 or 0 for rotating
            K: intrinsics matrix
            coeffs: 8-coeff distortion, in order p1, p2, k1, k2, p3, p4, p5, p6 (or whatever order open-cv has)
            w,h: int, heigh and with of image
    """
    img = cv2.imread(fp, cv2.IMREAD_UNCHANGED)
    if rc == -1:
        img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif rc == 1:
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

    map1, map2 = cv2.initUndistortRectifyMap(K, coeffs, None, K, (w, h), cv2.CV_32FC1)
    img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_NEAREST)

    return img


# Class for the dataset
class CondenseData(Dataset):
    def __init__(
            self,
            datadir,
            split='train',
            downsample=1.0
    ):

        self.img_wh = (
            int(1440 / downsample),
            int(2560 / downsample),
        )

        self.root_dir = datadir
        self.split = split
        with open(os.path.join(datadir, f"rotation_correction.json")) as f:
            self.rotation_correction = json.load(f)

        self.cam_infos = decompose_dataset(datadir, split=split)
        self.transform = T.ToTensor()
        # self.image_paths, self.image_poses, self.image_times = self.load_images_path(datadir, split)

    def viz_poses_for_KPlanes(self,):
        """
        Return shape N, 4, 4 in c2w form
        """
        from .ray_utils import stack_camera_dirs, get_rays
        import plotly.graph_objects as go
        fig = go.Figure()
        for cam_info in tqdm(self.cam_infos, total=len(self.cam_infos)):
            meta = self.cam_infos[cam_info]
            M = meta['M'].inverse()

            # Plot the camera poses
            intrinsics = Intrinsics(
                width=meta['W'], height=meta['H'], focal_x=meta['fx'], focal_y=meta['fy'], center_x=meta['cx'],
                center_y=meta['cy']
            )
            x = torch.tensor([[0], [0], [self.img_wh[1]], [self.img_wh[1]]]).squeeze(-1)
            y = torch.tensor([[0], [self.img_wh[0]], [self.img_wh[0]], [0]]).squeeze(-1)

            c2w = M

            camera_dirs = stack_camera_dirs(x,y,intrinsics, opengl_camera=False)
            rays_o, rays_d = get_rays(
                camera_dirs, c2w, ndc=False, ndc_near=1.0, intrinsics=intrinsics,
                normalize_rd=True)

            rays_o_np = rays_o.cpu().numpy()
            rays_d_np = rays_d.cpu().numpy()

            # Compute ray endpoints for visualization
            ray_length = 0.1  # Adjust visualization scale
            ray_endpoints = rays_o_np + (rays_d_np * ray_length)

            # Add rays to plotly figure
            for i in range(rays_o_np.shape[0]):
                fig.add_trace(go.Scatter3d(
                    x=[rays_o_np[i, 0], ray_endpoints[i, 0]],
                    y=[rays_o_np[i, 1], ray_endpoints[i, 1]],
                    z=[rays_o_np[i, 2], ray_endpoints[i, 2]],
                    mode='lines',
                    line=dict(color='blue', width=2),
                    showlegend=False
                ))

        fig.update_layout(
            title="Ray Visualization",
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z'
            )
        )
        fig.show()
        exit()

    def get_KPlanes_poses(self,):
        """
        Return shape N, 4, 4 in c2w form
        """
        from .ray_utils import stack_camera_dirs, get_rays
        images = []
        poses = []
        times = [] # -1 to 1

        if self.split == 'train':img_type = 'color_corrected'
        else: img_type = 'masks'

        for cam_info in tqdm(self.cam_infos, total=len(self.cam_infos)):
            meta = self.cam_infos[cam_info]
            M = meta['M'].inverse()
            fp = os.path.join(self.root_dir, f"{self.split}/{cam_info}/{img_type}/")

            time_max = len(os.listdir(fp))
            for img_fp in tqdm(os.listdir(fp), total=time_max):
                img_fp_ = os.path.join(fp, img_fp)
                img = Image.open(img_fp_)
                img = img.resize(self.img_wh, Image.LANCZOS)
                img = self.transform(img)
                images.append(img)
                poses.append(M)
                times.append(float(int(img_fp.split('.')[0]) / time_max)*2. -1.)

            # break
        return torch.stack(images, 0), torch.stack(poses), torch.tensor(times)

    def get_KPlanes_Intrinsics(self,):
        data = self.cam_infos[list(self.cam_infos.keys())[0]]
        return Intrinsics(
            width=data['W'], height=data['H'], focal_x=data['fx'], focal_y=data['fy'], center_x=data['cx'], center_y=data['cy']
        )

    def load_median_images(self):
        med_img_fp = os.path.join(self.root_dir, 'median_images')
        if os.path.exists(med_img_fp):
            print('Loading median images from {}'.format(med_img_fp))
            median_images = None
            for med_img in os.listdir(med_img_fp):
                median_images = torch.load(os.path.join(med_img_fp, f'{self.split}.pth'))
        else:
            print('No median images found at {} ... Generating median images'.format(med_img_fp))
            median_images = []
            direct = 'color_corrected' if self.split == 'train' else 'masks'
            for cam_info in tqdm(self.cam_infos, total=len(self.cam_infos)):
                fp = os.path.join(self.root_dir, f"{self.split}/{cam_info}/{direct}/")

                imgs = []
                for img_fp in tqdm(os.listdir(fp), total=len(os.listdir(fp))):
                    img_fp_ = os.path.join(fp, img_fp)

                    img = Image.open(img_fp_)
                    # img = img.resize(self.img_wh, Image.LANCZOS)
                    img = self.transform(img)
                    imgs.append(img)
                imgs = torch.stack(imgs, 0)
                med_img, _ = torch.median(imgs, dim=0)  # [h, w, 3]
                median_images.append(med_img)

            median_images = torch.stack(median_images, 0) # Save shape C, 3, H, W
            torch.save(median_images, os.path.join(med_img_fp, f'{self.split}.pth'))

        median_images = F.interpolate(median_images, size=(median_images.shape[2] // 4, median_images.shape[3] // 4), mode='bilinear', align_corners=False)

        median_images = median_images.permute(0, 2, 3, 1)

        return median_images


    def __len__(self):
        return len(self.image_paths)

    def load_pose(self, index):
        return self.image_poses[index]


class BaseClass:
    def __init__(self, datadir, settings):
        self.settings = settings

        # Initialise the train and test datasets (these will act as template data loaders for datasets)
        self.train_ds = CondenseData(datadir, split='train')
        self.test_ds = CondenseData(datadir, split='test')

        # If we need to undistort the data, we should do that
        print('Undistorting Images...')
        if settings['undistort']:
            self.test_ds.undistort_dataset()
            self.train_ds.undistort_dataset()

        print('... done')


