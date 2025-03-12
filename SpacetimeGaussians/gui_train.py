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
import torch
from random import randint
import random 
import sys 
import uuid
import time 
import json

import torchvision
import numpy as np 
import torch.nn.functional as F
import cv2
from kornia.geometry import translate
from tqdm import tqdm
import dearpygui.dearpygui as dpg


sys.path.append("./thirdparty/gaussian_splatting")

from thirdparty.gaussian_splatting.utils.loss_utils import l1_loss, ssim, l2_loss, rel_loss
from helper_train import getrenderpip, getmodel, getloss, controlgaussians, reloadhelper, trbfunction, setgtisint8, getgtisint8
from thirdparty.gaussian_splatting.scene import Scene
from argparse import Namespace
from thirdparty.gaussian_splatting.helper3dg import getparser, getrenderparts

from scipy.spatial.transform import Rotation as R
import math

from thirdparty.gaussian_splatting.renderer import  render_no_train

from image_metrics import psnr, ssim

def getWorld2View2(R, t, translate=np.array([.0, .0, .0]), scale=1.0):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    C2W = np.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = np.linalg.inv(C2W)
    return np.float32(Rt)


def getProjectionMatrix(znear, zfar, fovX, fovY):
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 1 / tanHalfFovX
    P[1, 1] = 1 / tanHalfFovY
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P


class MiniCam:
    def __init__(self, c2w, width, height, fovy, fovx, znear, zfar, time, from_training_view=False):
        # c2w (pose) should be in NeRF convention.

        self.image_width = width
        self.image_height = height
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.timestamp = time

        loaded = False
        if from_training_view:
            if type(c2w) == type({'type': 'dict'}):
                self.world_view_transform = c2w['world_view_transform']

                self.c2w = torch.linalg.inv(self.world_view_transform.cuda().transpose(0, 1))
                self.projection_matrix = c2w['projection_matrix']
                self.full_proj_transform = c2w['full_proj_transform']

                loaded = True

        if not loaded:
            self.c2w = c2w
            w2c = np.linalg.inv(c2w)
            # rectify...
            w2c[1:3, :3] *= -1
            w2c[:3, 3] *= -1

            self.world_view_transform = torch.tensor(w2c).transpose(0, 1).cuda().float()
            self.projection_matrix = (
                getProjectionMatrix(
                    znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy
                )
                .transpose(0, 1)
                .cuda().float()
            )
            self.full_proj_transform = self.world_view_transform @ self.projection_matrix

        self.camera_center = -torch.tensor(self.c2w[:3, 3]).cuda()

    def reset_extrinsic(self, R, T):
        self.world_view_transform = torch.tensor(getWorld2View2(R, T)).transpose(0, 1).cuda()
        self.full_proj_transform = (
            self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]

class OrbitCamera:
    def __init__(self, W, H, r=2, fov=60, near=0.01, far=100):
        self.W = W
        self.H = H
        self.radius = r  # camera distance from center

        self.fovy = 2 * np.arctan(np.tan(fov[0] / 2) * 0.25)
        self.fovy = np.deg2rad(self.fovy)  # deg 2 rad
        self.near = near
        self.far = far
        self.center = np.array([0, 0, 0], dtype=np.float32)  # look at this point
        self.rot = R.from_matrix(np.array([[1., 0., 0., ],
                                           [0., 0., -1.],
                                           [0., 1., 0.]]))
        self.up = np.array([0, 1, 0], dtype=np.float32)  # need to be normalized!
        self.side = np.array([1, 0, 0], dtype=np.float32)

    @property
    def fovx(self):
        return 2 * np.arctan(np.tan(self.fovy / 2) * self.W / self.H)

    @property
    def campos(self):
        return self.pose[:3, 3]

    # pose (c2w)
    @property
    def pose(self):
        # first move camera to radius
        res = np.eye(4, dtype=np.float32)
        res[2, 3] = self.radius  # opengl convention...
        # rotate
        rot = np.eye(4, dtype=np.float32)
        rot[:3, :3] = self.rot.as_matrix()
        res = rot @ res
        # translate
        res[:3, 3] -= self.center
        return res

    # view (w2c)
    @property
    def view(self):
        return np.linalg.inv(self.pose)

    # projection (perspective)
    @property
    def perspective(self):
        y = np.tan(self.fovy / 2)
        aspect = self.W / self.H
        return np.array(
            [
                [1 / (y * aspect), 0, 0, 0],
                [0, -1 / y, 0, 0],
                [
                    0,
                    0,
                    -(self.far + self.near) / (self.far - self.near),
                    -(2 * self.far * self.near) / (self.far - self.near),
                ],
                [0, 0, -1, 0],
            ],
            dtype=np.float32,
        )

    # intrinsics
    @property
    def intrinsics(self):
        focal = self.H / (2 * np.tan(self.fovy / 2))
        return np.array([focal, focal, self.W // 2, self.H // 2], dtype=np.float32)

    @property
    def mvp(self):
        return self.perspective @ np.linalg.inv(self.pose)  # [4, 4]

    def orbit(self, dx, dy):
        # rotate along camera up/side axis!
        side = self.rot.as_matrix()[:3, 0]
        up = self.rot.as_matrix()[:3, 1]
        rotvec_x = up * np.radians(-0.05 * dx)
        rotvec_y = side * np.radians(-0.05 * dy)
        self.rot = R.from_rotvec(rotvec_x) * R.from_rotvec(rotvec_y) * self.rot

    def scale(self, delta):
        self.radius *= 1.1 ** (-delta)

    def pan(self, dx, dy, dz=0):
        # pan in camera coordinate system (careful on the sensitivity!)
        self.center += 0.0001 * self.rot.as_matrix()[:3, :3] @ np.array([-dx, -dy, dz])

    def get_proj_matrix(self):
        tanHalfFovY = math.tan((self.fovy / 2))
        tanHalfFovX = math.tan((self.fovx / 2))

        top = tanHalfFovY * self.near
        bottom = -top
        right = tanHalfFovX * self.near
        left = -right

        P = torch.zeros(4, 4)

        z_sign = 1.0

        P[0, 0] = 2.0 * self.near / (right - left)
        P[1, 1] = 2.0 * self.near / (top - bottom)
        P[0, 2] = (right + left) / (right - left)
        P[1, 2] = (top + bottom) / (top - bottom)
        P[3, 2] = z_sign
        P[2, 2] = z_sign * self.far / (self.far - self.near)
        P[2, 3] = -(self.far * self.near) / (self.far - self.near)
        return P

# Going from 3x3 rotation matrix to quaternion values
def R_to_q(R, eps=1e-8):  # [B,3,3]
    # https://en.wikipedia.org/wiki/Rotation_matrix#Quaternion
    # FIXME: this function seems a bit problematic, need to double-check
    row0, row1, row2 = R.unbind(dim=-2)
    R00, R01, R02 = row0.unbind(dim=-1)
    R10, R11, R12 = row1.unbind(dim=-1)
    R20, R21, R22 = row2.unbind(dim=-1)
    t = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
    r = (1 + t + eps).sqrt()
    qa = 0.5 * r
    qb = (R21 - R12).sign() * 0.5 * (1 + R00 - R11 - R22 + eps).sqrt()
    qc = (R02 - R20).sign() * 0.5 * (1 - R00 + R11 - R22 + eps).sqrt()
    qd = (R10 - R01).sign() * 0.5 * (1 - R00 - R11 + R22 + eps).sqrt()
    q = torch.stack([qa, qb, qc, qd], dim=-1)

    for i, qi in enumerate(q):
        if torch.isnan(qi).any():
            K = torch.stack([torch.stack([R00 - R11 - R22, R10 + R01, R20 + R02, R12 - R21], dim=-1),
                             torch.stack([R10 + R01, R11 - R00 - R22, R21 + R12, R20 - R02], dim=-1),
                             torch.stack([R20 + R02, R21 + R12, R22 - R00 - R11, R01 - R10], dim=-1),
                             torch.stack([R12 - R21, R20 - R02, R01 - R10, R00 + R11 + R22], dim=-1)], dim=-2) / 3.0
            K = K[i]
            eigval, eigvec = torch.linalg.eigh(K)
            V = eigvec[:, eigval.argmax()]
            q[i] = torch.stack([V[3], V[0], V[1], V[2]])
    return q

def hash_cams(cam):
    return cam.camera_center.sum().item()


class GaussianCameraModel:

    def get_camera_stuff(self, cam_list):
        xyzs = []
        qs = []

        hash_camlist = []

        # For each camera in list get the camera position and rotation
        for i, cam in enumerate(cam_list):
            # Hash the camera (i.e. avoid displaying cameras with the same R and T)
            cam_hash = hash_cams(cam)
            # We do this but summing the camera center points (sure it seems silly but this is realistically fine....(I think)
            if cam_hash not in hash_camlist:
                hash_camlist.append(cam_hash)

                for xyz in self.cam_xyzs:
                    T = cam.camera_center + xyz.cuda()
                    w2c = cam.world_view_transform.transpose(0, 1)
                    R = w2c[:3, :3].transpose(0, 1)
                    T = torch.matmul(R, xyz.unsqueeze(-1).cuda()).squeeze(-1) + cam.camera_center

                    q = R_to_q(R)

                    xyzs.append(T.unsqueeze(0))
                    qs.append(q.unsqueeze(0))
            # else:
            #     break

        self.qs = torch.cat(qs, dim=0).cuda()
        self.xyzs = torch.cat(xyzs, dim=0).cuda()

        self.xyzs.requires_grad = True
        self.qs.requires_grad = True

    def get_cam_model(self, cam, W, H):
        w = float(W / 2.)
        h = float(H / 2.)

        x_num_pts = 11

        w2c = cam.world_view_transform.transpose(0, 1)
        R = w2c[:3, :3].transpose(0, 1)
        xyzs = []
        qs = []

        if w > h:
            x_size = float(4. * x_num_pts)
            y_size = float(3. * x_num_pts)
        elif w < h:
            x_size = float(3. * x_num_pts)
            y_size = float(4. * x_num_pts)
        else:
            x_size = float(3. * x_num_pts)
            y_size = float(3. * x_num_pts)

        # Line in +X (Right)
        for i in range(0, x_num_pts + 1):
            x = float(i - (x_num_pts) / 2.) / x_size
            T = torch.tensor([x, +0.2, 0.])
            q = R_to_q(R)

            xyzs.append(T.unsqueeze(0))
            qs.append(q.unsqueeze(0))

        for i in range(0, x_num_pts + 1):
            x = float(i - (x_num_pts) / 2.) / x_size
            T = torch.tensor([x, -0.2, 0.])
            q = R_to_q(R)

            xyzs.append(T.unsqueeze(0))
            qs.append(q.unsqueeze(0))

        # Line in -Z (Back)
        for i in range(0, 5):
            T = torch.tensor([0., 0., - float(i / 10.)])
            q = R_to_q(R)

            xyzs.append(T.unsqueeze(0))
            qs.append(q.unsqueeze(0))

        # Line in +Y (Down) Show direction
        T = torch.tensor([-0.1, 0., 0.])
        q = R_to_q(R)
        xyzs.append(T.unsqueeze(0))
        qs.append(q.unsqueeze(0))

        # Show -x -y for the topleft hand image
        T = torch.tensor([0., -0.1, 0.])
        q = R_to_q(R)
        xyzs.append(T.unsqueeze(0))
        qs.append(q.unsqueeze(0))

        self.cam_qs = torch.cat(qs, dim=0)
        self.cam_xyzs = torch.cat(xyzs, dim=0)

    def __init__(self, cam_list, W, H) -> None:

        max_i = 1
        if len(cam_list) > 100: max_i = 300


        cam = cam_list[0]
        cam_list = [cam_list[i] for i in range(len(cam_list)) if i % max_i == 0]
        self.get_cam_model(cam, W, H)

        self.get_camera_stuff(cam_list)

        cam_centers = []
        cam_hash = []

        for i, cam in enumerate(cam_list):
            T = cam.camera_center
            if str(T) not in cam_hash:
                cam_centers.append(T)
                cam_hash.append(str(T))
        tensor_stack = torch.stack(cam_centers)
        self.cam_center = torch.mean(tensor_stack, dim=0)

        # Blob scale NOT camera scale
        self.scale = torch.tensor([0.001, 0.001, 0.001]).cuda()


class GUI:

    def __init__(self, dataset, opt, pipe, saving_iterations, debug_from, densify=0, duration=50, rgbfunction="rgbv1", rdpip="v2"):

        with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
            cfg_log_f.write(str(Namespace(**vars(args))))

        self.render, self.GRsetting, self.GRzer = getrenderpip(rdpip)

        print("use model {}".format(dataset.model))
        GaussianModel = getmodel(dataset.model) # gmodel, gmodelrgbonly

        self.gaussians = GaussianModel(dataset.sh_degree, rgbfunction)
        self.gaussians.trbfslinit = -1*opt.trbfslinit #
        self.gaussians.preprocesspoints = opt.preprocesspoints
        self.gaussians.addsphpointsscale = opt.addsphpointsscale
        self.gaussians.raystart = opt.raystart

        self.opt = opt
        self.pipe = pipe
        self.saving_iterations = saving_iterations
        self.debug_from = debug_from
        self.densify = densify

        self.rbfbasefunction = trbfunction
        self.scene = Scene(dataset, self.gaussians, duration=duration, loader=dataset.loader)
        self.duration = duration

        currentxyz = self.gaussians._xyz
        maxx, maxy, maxz = torch.amax(currentxyz[:,0]), torch.amax(currentxyz[:,1]), torch.amax(currentxyz[:,2])# z wrong...
        minx, miny, minz = torch.amin(currentxyz[:,0]), torch.amin(currentxyz[:,1]), torch.amin(currentxyz[:,2])


        if os.path.exists(self.opt.prevpath):
            print("load from " + self.opt.prevpath)
            reloadhelper(self.gaussians, self.opt, maxx, maxy, maxz,  minx, miny, minz)



        self.maxbounds = [maxx, maxy, maxz]
        self.minbounds = [minx, miny, minz]

        self.gaussians.training_setup(self.opt)

        numchannel = 9
        bg_color = [1, 1, 1] if dataset.white_background else [0 for i in range(numchannel)]
        self.background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        # DPG stuff
        print('DPG loading ...')
        sample  = self.scene.train_cameras.dataset[0]
        self.W, self.H = sample.image_width, sample.image_height
        self.fov = (sample.FoVy, sample.FoVx)


        self.W = int(self.W / 4)
        self.H = int(self.H / 4)

        self.fovy = self.fov[0]
        self.cam = OrbitCamera(self.W, self.H, r=60., fov=self.fov)
        self.store_orbit = self.cam
        self.mode = "rgb"
        self.buffer_image = np.ones((self.W, self.H, 3), dtype=np.float32)
        self.time = 0.
        self.show_scene = True
        self.show_depth = False
        self.show_cameras = True

        self.training_cams_pc = GaussianCameraModel(self.scene.train_cameras.dataset, self.W, self.H)

        dpg.create_context()
        self.register_dpg()

    def __del__(self):
        dpg.destroy_context()

    def register_dpg(self):
        ### register texture
        with dpg.texture_registry(show=False):
            dpg.add_raw_texture(
                self.W,
                self.H,
                self.buffer_image,
                format=dpg.mvFormat_Float_rgb,
                tag="_texture",
            )

        ### register window
        # the rendered image, as the primary window
        with dpg.window(
                tag="_primary_window",
                width=self.W,
                height=self.H,
                pos=[0, 0],
                no_move=True,
                no_title_bar=True,
                no_scrollbar=True,
        ):
            # add the texture
            dpg.add_image("_texture")

        # dpg.set_primary_window("_primary_window", True)

        # control window
        with dpg.window(
                label="Control",
                tag="_control_window",
                width=400,
                height=self.H,
                pos=[self.W, 0],
                no_move=True,
                no_title_bar=True,
        ):
            # button theme
            with dpg.theme() as theme_button:
                with dpg.theme_component(dpg.mvButton):
                    dpg.add_theme_color(dpg.mvThemeCol_Button, (23, 3, 18))
                    dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (51, 3, 47))
                    dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (83, 18, 83))
                    dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 5)
                    dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 3, 3)

            # timer stuff
            with dpg.group(horizontal=True):
                dpg.add_text("Infer time: ")
                dpg.add_text("no data", tag="_log_infer_time")

            with dpg.group():
                dpg.add_text("Training info: ")
                dpg.add_text("no data", tag="_log_iter")
                dpg.add_text("no data", tag="_log_loss")
                dpg.add_text("no data", tag="_log_depth")
                dpg.add_text("no data", tag="_log_points")

            with dpg.collapsing_header(label="Testing info:", default_open=True):
                dpg.add_text("no data", tag="_log_psnr_test")
                dpg.add_text("no data", tag="_log_ssim")

            # rendering options
            with dpg.collapsing_header(label="Rendering", default_open=True):
                def callback_toggle_show_depth(sender):
                    self.show_depth = not self.show_depth

                with dpg.group(horizontal=True):
                    dpg.add_button(label="Depth", callback=callback_toggle_show_depth)

                def callback_show_max_radius(sender):
                    self.need_update = True

                dpg.add_slider_float(
                    label="Show Radial Distance",
                    default_value=2.,
                    max_value=20.,
                    min_value=0.,
                    callback=callback_show_max_radius,
                )

                def callback_speed_control(sender):
                    self.time = dpg.get_value(sender)
                    self.need_update = True

                dpg.add_slider_float(
                    label="Time",
                    default_value=0.,
                    max_value=1.,
                    min_value=0.,
                    callback=callback_speed_control,
                )

                def callback_toggle_show_scene(sender):
                    self.show_scene = not self.show_scene

                def callback_toggle_show_cameras(sender):
                    self.show_cameras = not self.show_cameras

                with dpg.group(horizontal=True):
                    dpg.add_button(label="Show/Hide Scene", callback=callback_toggle_show_scene)
                    dpg.add_button(label="Show/Hide Cameras", callback=callback_toggle_show_cameras)

        def callback_set_mouse_loc(sender, app_data):
            if not dpg.is_item_focused("_primary_window"):
                return
            self.mouse_loc = np.array(app_data)

        ### register camera handler

        def callback_camera_drag_rotate_or_draw_mask(sender, app_data):
            if not dpg.is_item_focused("_primary_window"):
                return

            dx = app_data[1]
            dy = app_data[2]

            self.cam.orbit(dx, dy)

            self.need_update = True

        def callback_camera_wheel_scale(sender, app_data):
            if not dpg.is_item_focused("_primary_window"):
                return

            delta = app_data

            self.cam.scale(delta)

            self.need_update = True

        def callback_camera_drag_pan(sender, app_data):
            if not dpg.is_item_focused("_primary_window"):
                return

            dx = app_data[1]
            dy = app_data[2]

            self.cam.pan(dx, dy)
            self.need_update = True

        with dpg.handler_registry():
            # for camera moving
            dpg.add_mouse_drag_handler(
                button=dpg.mvMouseButton_Left,
                callback=callback_camera_drag_rotate_or_draw_mask,
            )
            dpg.add_mouse_wheel_handler(callback=callback_camera_wheel_scale)
            dpg.add_mouse_drag_handler(
                button=dpg.mvMouseButton_Middle, callback=callback_camera_drag_pan
            )

            dpg.add_mouse_move_handler(callback=callback_set_mouse_loc)

        dpg.create_viewport(
            title="WavePlanes",
            width=self.W + 400,
            height=self.H + (45 if os.name == "nt" else 0),
            resizable=False,
        )

        ### global theme
        with dpg.theme() as theme_no_padding:
            with dpg.theme_component(dpg.mvAll):
                # set all padding to 0 to avoid scroll bar
                dpg.add_theme_style(
                    dpg.mvStyleVar_WindowPadding, 0, 0, category=dpg.mvThemeCat_Core
                )
                dpg.add_theme_style(
                    dpg.mvStyleVar_FramePadding, 0, 0, category=dpg.mvThemeCat_Core
                )
                dpg.add_theme_style(
                    dpg.mvStyleVar_CellPadding, 0, 0, category=dpg.mvThemeCat_Core
                )

        dpg.bind_item_theme("_primary_window", theme_no_padding)

        dpg.setup_dearpygui()

        dpg.show_viewport()

    @torch.no_grad()
    def viewer_step(self):
        image = self.test()
        buffer_image = torch.nn.functional.interpolate(
            image.unsqueeze(0),
            size=(self.H, self.W),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)



        self.buffer_image = (
            buffer_image.permute(1, 2, 0)
            .contiguous()
            .clamp(0, 1)
            .contiguous()
            .detach()
            .cpu()
            .numpy()
        )

        # ender.record()
        self.need_update = True

        # t = starter.elapsed_time(ender)

        buffer_image = self.buffer_image
        # dpg.set_value("_log_infer_time", f"{t:.4f}ms ({int(1000/t)} FPS)")
        dpg.set_value(
            "_texture", buffer_image
        )  # buffer must be contiguous, else seg fault!

        dpg.render_dearpygui_frame()

    def test(self):
        with torch.no_grad():
            PSNR = 0.
            SSIM = 0.
            test_data = self.scene.get_test_cam_list()
            tot_imgs = len(test_data)
            for viewpoint_cam in test_data:
                render_pkg = self.render(viewpoint_cam, self.gaussians, self.pipe, self.background, override_color=None,
                                         basicfunction=self.rbfbasefunction, GRsetting=self.GRsetting, GRzer=self.GRzer)
                image, viewspace_point_tensor, visibility_filter, radii = getrenderparts(render_pkg)

                if getgtisint8():
                    gt_image = viewpoint_cam.original_image.cuda().float() / 255.0
                else:
                    # cast float on cuda will introduce gradient, so cast first then to cuda. at the cost of i/o
                    gt_image = viewpoint_cam.original_image.float().cuda()
                if self.opt.gtmask:  # for training with undistorted immerisve image, masking black pixels in undistorted image.
                    mask = torch.sum(gt_image, dim=0) == 0
                    mask = mask.float()
                    image = image * (1 - mask) + gt_image * (mask)

                gt_image = gt_image[:3, ...]

                # Image metrics PSNR and SSIM
                psnr_ = psnr(gt_image, image)
                PSNR+= psnr_
                SSIM += ssim(image.unsqueeze(0), gt_image)
        print(f"PSNR: {PSNR / tot_imgs}")
        print(f"SSIM: {SSIM / tot_imgs}")

        return image

    def train(self):
        flag = 0
        if self.gaussians.ts is None:
            H, W = self.H*4, self.W*4
            self.gaussians.ts = torch.ones(1, 1, H, W).cuda()

        self.scene.recordpoints(0, "start training")

        flagems = 0
        emscnt = 0
        lossdiect = {}
        ssimdict = {}
        self.depthdict = {}
        validdepthdict = {}
        emsstartfromiterations = self.opt.emsstart
        with torch.no_grad():
            viewpointset = self.scene.get_train_cam_list(0)
            for viewpoint_cam in viewpointset:
                render_pkg = self.render(viewpoint_cam, self.gaussians, self.pipe, self.background, override_color=None,
                                         basicfunction=self.rbfbasefunction, GRsetting=self.GRsetting, GRzer=self.GRzer)

                _, depthH, depthW = render_pkg["depth"].shape

                depth = render_pkg["depth"]
                slectemask = depth != 15.0

                validdepthdict[viewpoint_cam.image_name] = torch.median(depth[slectemask]).item()
                self.depthdict[viewpoint_cam.image_name] = torch.amax(depth[slectemask]).item()

        if self.densify == 1 or self.densify == 2:
            zmask = self.gaussians._xyz[:, 2] < 4.5
            self.gaussians.prune_points(zmask)
            torch.cuda.empty_cache()

        iter_start = torch.cuda.Event(enable_timing=True)
        iter_end = torch.cuda.Event(enable_timing=True)

        ema_loss_for_log = 0.0
        progress_bar = tqdm(range(0, self.opt.iterations), desc="Training progress")
        first_iter = 1

        selectedlength = 2
        lasterems = 0
        gtisint8 = getgtisint8()

        print('Starting training')
        for iteration in range(first_iter, self.opt.iterations + 1):


            if iteration ==  self.opt.emsstart:
                flagems = 1 # start ems

            iter_start.record()
            self.gaussians.update_learning_rate(iteration)

            if (iteration - 1) == self.densify:
                self.pipe.debug = True
            if self.gaussians.rgbdecoder is not None:
                self.gaussians.rgbdecoder.train()

            if self.opt.batch > 1:

                self.gaussians.zero_gradient_cache()
                timeindex = randint(0, self.duration-1) # 0 to 49
                viewpointset = self.scene.get_train_cam_list(timeindex)
                camindex = random.sample(viewpointset, self.opt.batch)

                for i in range(self.opt.batch):
                    viewpoint_cam = camindex[i]
                    render_pkg = self.render(viewpoint_cam, self.gaussians, self.pipe, self.background,  override_color=None,  basicfunction=self.rbfbasefunction, GRsetting=self.GRsetting, GRzer=self.GRzer)
                    image, viewspace_point_tensor, visibility_filter, radii = getrenderparts(render_pkg)

                    if gtisint8:
                        gt_image = viewpoint_cam.original_image.cuda().float()/255.0
                    else:
                        # cast float on cuda will introduce gradient, so cast first then to cuda. at the cost of i/o
                        gt_image = viewpoint_cam.original_image.float().cuda()
                    if self.opt.gtmask: # for training with undistorted immerisve image, masking black pixels in undistorted image.
                        mask = torch.sum(gt_image, dim=0) == 0
                        mask = mask.float()
                        image = image * (1- mask) +  gt_image * (mask)

                    # Mul mask with gt - so that we get a zero background
                    gt_image = gt_image[:3, ...] *  gt_image[-1, ...]

                    if self.opt.reg == 2:
                        Ll1 = l2_loss(image, gt_image)
                        loss = Ll1
                    elif self.opt.reg == 3:
                        Ll1 = rel_loss(image, gt_image)
                        loss = Ll1
                    else:
                        Ll1 = l1_loss(image, gt_image)
                        loss = getloss(self.opt, Ll1, ssim, image, gt_image, self.gaussians, radii)

                    if flagems == 1:
                        if viewpoint_cam.image_name not in lossdiect:
                            lossdiect[viewpoint_cam.image_name] = loss.item()
                            ssimdict[viewpoint_cam.image_name] = ssim(image.clone().detach(), gt_image.clone().detach()).item()


                    loss.backward()
                    self.gaussians.cache_gradient()
                    self.gaussians.optimizer.zero_grad(set_to_none = True)#


                if flagems == 1 and len(lossdiect.keys()) == len(viewpointset):
                    # sort dict by value
                    orderedlossdiect = sorted(ssimdict.items(), key=lambda item: item[1], reverse=False) # ssimdict lossdiect
                    flagems = 2
                    selectviewslist = []
                    selectviews = {}
                    for idx, pair in enumerate(orderedlossdiect):
                        viewname, lossscore = pair
                        ssimscore = ssimdict[viewname]
                        if ssimscore < 0.91: # avoid large ssim
                            selectviewslist.append((viewname, "rk"+ str(idx) + "_ssim" + str(ssimscore)[0:4]))
                    if len(selectviewslist) < 2 :
                        selectviews = []
                    else:
                        selectviewslist = selectviewslist[:2]
                        for v in selectviewslist:
                            selectviews[v[0]] = v[1]

                    selectedlength = len(selectviews)

                iter_end.record()
                self.gaussians.set_batch_gradient(self.opt.batch)
                 # note we retrieve the correct gradient except the mask
            else:
                raise NotImplementedError("Batch size 1 is not supported")

            if iteration % 100 == 0 and iteration > 1800:
                self.viewer_step()

            with torch.no_grad():
                # Progress bar
                ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
                if iteration % 10 == 0:
                    progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                    progress_bar.update(10)
                if iteration == self.opt.iterations:
                    progress_bar.close()

                if (iteration in self.saving_iterations):
                    print("\n[ITER {}] Saving self.gaussians".format(iteration))
                    self.scene.save(iteration)

                # Densification and pruning here

                if iteration < self.opt.densify_until_iter :
                    self.gaussians.max_radii2D[visibility_filter] = torch.max(self.gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                    self.gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)
                flag = controlgaussians(self.opt, self.gaussians, self.densify, iteration, self.scene,  visibility_filter, radii, viewspace_point_tensor, flag,  traincamerawithdistance=None, maxbounds=self.maxbounds,minbounds=self.minbounds)

                # guided sampling step
                if iteration > emsstartfromiterations and flagems == 2 and emscnt < selectedlength and viewpoint_cam.image_name in selectviews and (iteration - lasterems > 100): #["camera_0002"] :#selectviews :  #["camera_0002"]:
                    selectviews.pop(viewpoint_cam.image_name) # remove sampled cameras
                    emscnt += 1
                    lasterems = iteration
                    ssimcurrent = ssim(image.detach(), gt_image.detach()).item()
                    self.scene.recordpoints(iteration, "ssim_" + str(ssimcurrent))
                    # some scenes' strcture is already good, no need to add more points
                    if ssimcurrent < 0.88:
                        imageadjust = image /(torch.mean(image)+0.01) #
                        gtadjust = gt_image / (torch.mean(gt_image)+0.01)
                        diff = torch.abs(imageadjust   - gtadjust)
                        diff = torch.sum(diff,        dim=0) # h, w
                        diff_sorted, _ = torch.sort(diff.reshape(-1))
                        numpixels = diff.shape[0] * diff.shape[1]
                        threshold = diff_sorted[int(numpixels*self.opt.emsthr)].item()
                        outmask = diff > threshold#
                        kh, kw = 16, 16 # kernel size
                        dh, dw = 16, 16 # stride
                        idealh, idealw = int(image.shape[1] / dh  + 1) * kw, int(image.shape[2] / dw + 1) * kw # compute padding
                        outmask = torch.nn.functional.pad(outmask, (0, idealw - outmask.shape[1], 0, idealh - outmask.shape[0]), mode='constant', value=0)
                        patches = outmask.unfold(0, kh, dh).unfold(1, kw, dw)
                        dummypatch = torch.ones_like(patches)
                        patchessum = patches.sum(dim=(2,3))
                        patchesmusk = patchessum  >  kh * kh * 0.85
                        patchesmusk = patchesmusk.unsqueeze(2).unsqueeze(3).repeat(1,1,kh,kh).float()
                        patches = dummypatch * patchesmusk

                        depth = render_pkg["depth"]
                        depth = depth.squeeze(0)
                        idealdepthh, idealdepthw = int(depth.shape[0] / dh  + 1) * kw, int(depth.shape[1] / dw + 1) * kw # compute padding for depth

                        depth = torch.nn.functional.pad(depth, (0, idealdepthw - depth.shape[1], 0, idealdepthh - depth.shape[0]), mode='constant', value=0)

                        depthpaches = depth.unfold(0, kh, dh).unfold(1, kw, dw)
                        dummydepthpatches =  torch.ones_like(depthpaches)
                        a,b,c,d = depthpaches.shape
                        depthpaches = depthpaches.reshape(a,b,c*d)
                        mediandepthpatch = torch.median(depthpaches, dim=(2))[0]
                        depthpaches = dummydepthpatches * (mediandepthpatch.unsqueeze(2).unsqueeze(3))
                        unfold_depth_shape = dummydepthpatches.size()
                        output_depth_h = unfold_depth_shape[0] * unfold_depth_shape[2]
                        output_depth_w = unfold_depth_shape[1] * unfold_depth_shape[3]

                        patches_depth_orig = depthpaches.view(unfold_depth_shape)
                        patches_depth_orig = patches_depth_orig.permute(0, 2, 1, 3).contiguous()
                        patches_depth = patches_depth_orig.view(output_depth_h, output_depth_w).float() # 1 for error, 0 for no error

                        depth = patches_depth[:render_pkg["depth"].shape[1], :render_pkg["depth"].shape[2]]
                        depth = depth.unsqueeze(0)


                        midpatch = torch.ones_like(patches)


                        for i in range(0, kh,  2):
                            for j in range(0, kw, 2):
                                midpatch[:,:, i, j] = 0.0

                        centerpatches = patches * midpatch

                        unfold_shape = patches.size()
                        patches_orig = patches.view(unfold_shape)
                        centerpatches_orig = centerpatches.view(unfold_shape)

                        output_h = unfold_shape[0] * unfold_shape[2]
                        output_w = unfold_shape[1] * unfold_shape[3]
                        patches_orig = patches_orig.permute(0, 2, 1, 3).contiguous()
                        centerpatches_orig = centerpatches_orig.permute(0, 2, 1, 3).contiguous()
                        centermask = centerpatches_orig.view(output_h, output_w).float() # H * W  mask, # 1 for error, 0 for no error
                        centermask = centermask[:image.shape[1], :image.shape[2]] # reverse back

                        errormask = patches_orig.view(output_h, output_w).float() # H * W  mask, # 1 for error, 0 for no error
                        errormask = errormask[:image.shape[1], :image.shape[2]] # reverse back

                        H, W = centermask.shape

                        offsetH = int(H/10)
                        offsetW = int(W/10)

                        centermask[0:offsetH, :] = 0.0
                        centermask[:, 0:offsetW] = 0.0

                        centermask[-offsetH:, :] = 0.0
                        centermask[:, -offsetW:] = 0.0


                        depth = render_pkg["depth"]
                        depthmap = torch.cat((depth, depth, depth), dim=0)
                        invaliddepthmask = depth == 15.0

                        pathdir = self.scene.model_path + "/ems_" + str(emscnt-1)
                        if not os.path.exists(pathdir):
                            os.makedirs(pathdir)

                        depthmap = depthmap / torch.amax(depthmap)
                        invalideptmap = torch.cat((invaliddepthmask, invaliddepthmask, invaliddepthmask), dim=0).float()


                        torchvision.utils.save_image(gt_image, os.path.join(pathdir,  "gt" + str(iteration) + ".png"))
                        torchvision.utils.save_image(image, os.path.join(pathdir,  "render" + str(iteration) + ".png"))
                        torchvision.utils.save_image(depthmap, os.path.join(pathdir,  "depth" + str(iteration) + ".png"))
                        torchvision.utils.save_image(invalideptmap, os.path.join(pathdir,  "indepth" + str(iteration) + ".png"))


                        badindices = centermask.nonzero()
                        diff_sorted , _ = torch.sort(depth.reshape(-1))
                        N = diff_sorted.shape[0]
                        mediandepth = int(0.7 * N)
                        mediandepth = diff_sorted[mediandepth]

                        depth = torch.where(depth>mediandepth, depth,mediandepth )

                        totalNnewpoints = self.gaussians.addgaussians(badindices, viewpoint_cam, depth, gt_image, numperay=self.opt.farray,ratioend=self.opt.rayends,  depthmax=self.depthdict[viewpoint_cam.image_name], shuffle=(self.opt.shuffleems != 0))

                        gt_image = gt_image * errormask
                        image = render_pkg["render"] * errormask

                        self.scene.recordpoints(iteration, "after addpointsbyuv")

                        torchvision.utils.save_image(gt_image, os.path.join(pathdir,  "maskedudgt" + str(iteration) + ".png"))
                        torchvision.utils.save_image(image, os.path.join(pathdir,  "maskedrender" + str(iteration) + ".png"))
                        visibility_filter = torch.cat((visibility_filter, torch.zeros(totalNnewpoints).cuda(0).bool()), dim=0)
                        visibility_filter = visibility_filter.bool()
                        radii = torch.cat((radii, torch.zeros(totalNnewpoints).cuda(0).int()), dim=0)
                        viewspace_point_tensor = torch.cat((viewspace_point_tensor, torch.zeros(totalNnewpoints, 3).cuda(0)), dim=0)



                # Optimizer step
                if iteration < self.opt.iterations:
                    self.gaussians.optimizer.step()

                    self.gaussians.optimizer.zero_grad(set_to_none = True)



if __name__ == "__main__":
    

    args, lp_extract, op_extract, pp_extract = getparser()
    setgtisint8(op_extract.gtisint8)

    gui = GUI(lp_extract, op_extract, pp_extract, args.save_iterations, args.debug_from, densify=args.densify, duration=args.duration, rgbfunction=args.rgbfunction, rdpip=args.rdpip)

    gui.train()
    # All done
    print("\nTraining complete.")
