import os
from torch.utils.data import Dataset
import json
import torch
import cv2
import numpy as np
import shutil


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

        R_m, T_m = default_deform_tracking(cap_area_config, 'cpu')
        M_m = torch.eye(4)
        M_m[:3, :3] = torch.tensor(R_m).view((3, 3))
        M_m[:3, 3] = torch.tensor(T_m).view((3, 1))[:, 0]

        # Project depth transform onto capture areas transform
        M_d = M_depth.inverse() @ M_m.inverse()
        T_d = M_d[:3, 3].numpy()
        R_d = M_d[:3, :3].numpy()

        """Color cameras are slightly shifted and rotated w.r.t depth cameras
        """
        M_col = torch.eye(4)
        M_col[:3, :3] = torch.tensor(col2depth_ex['orientation']).view((3, 3)).mT
        M_col[:3, 3] = torch.tensor(col2depth_ex['translation']).view((3, 1))[:, 0]

        # Project color transform onto depth transform
        M = M_col.inverse() @ M_d
        T = M[:3, 3].numpy()
        R = M[:3, :3].numpy()

        # Note that these parameters are in World space (I think...lol at least im not doing 4-D projective geometry (mdr))
        #   so they correspond to c2w transform in the form of (r.T @ (x-t).T).T
        #   and the w2c in the form of (r @ x.T).T + t
        poses[c] = {
            'name':c,
            'H': meta['colour_intrinsics']['height'], 'W': meta['colour_intrinsics']['width'],
            'fx': meta['colour_intrinsics']['fx'],
            'fy': meta['colour_intrinsics']['fy'],
            'R': R,
            'T': T,
            'cx': meta['colour_intrinsics']['ppx'],
            'cy': meta['colour_intrinsics']['ppy'],
            'distortion': color_distortion,
            'depth_params': {
                'H': meta['depth_intrinsics']['height'],
                'W': meta['depth_intrinsics']['width'],
                'R': R_d,
                'T': T_d,
                'fx': meta['depth_intrinsics']['fx'],
                'fy': meta['depth_intrinsics']['fy'],
                'cx': meta['depth_intrinsics']['ppx'],
                'cy': meta['depth_intrinsics']['ppy'],
                'distortion': depth_distortion

            }
        }

    """Final note (I promise):
        The preprocessing is NOT complete until we rotate the images correctly (yes they need to be rotated). 
        I haven't baked this into the transforms due to issues with mis-alignment in the RGB and DEPTH cameras (I'm probably doing something wrong
        becasuse I don't think this should should be an issue). My current fix is simply to rotate all images, including depth, and forget
        I even had this problem...
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
    ):

        self.root_dir = datadir
        with open(os.path.join(datadir, f"rotation_correction.json")) as f:
            self.rotation_correction = json.load(f)

        self.cam_infos = decompose_dataset(datadir, split=split)

        self.image_paths, self.image_poses, self.image_times = self.load_images_path(datadir, split)


    def load_images_path(self, cam_folder, split):
        image_paths = []
        image_poses = []
        image_times = []

        for cam_info in self.cam_infos:
            meta = self.cam_infos[cam_info]
            R = meta['R']
            T = meta['T']
            fp = os.path.join(cam_folder, f"{split}/{cam_info}/color/")

            time_max = len(os.listdir(fp))
            for img_fp in os.listdir(fp):
                img_fp_ = os.path.join(fp, img_fp)

                image_paths.append(img_fp_)
                image_poses.append((R, T))
                image_times.append(float(int(img_fp.split('.')[0]) / time_max))

        return image_paths, image_poses, image_times

    def __len__(self):
        return len(self.image_paths)

    def load_pose(self, index):
        return self.image_poses[index]


class BaseClass:
    def __init__(self, datadir):
        # Initialise the train and test datasets (these will act as template data loaders for datasets)
        self.train_ds = CondenseData(datadir, split='train')
        self.test_ds = CondenseData(datadir, split='test')


