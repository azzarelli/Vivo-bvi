import os
import time
import torch
import math
import numpy as np

from scipy.spatial.transform import Rotation as R

# from scene.cameras import MiniCam


import numpy as np
import random
import os, sys
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim, l2_loss, lpips_loss
from pytorch_msssim import ms_ssim

from gaussian_renderer import render
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams, ModelHiddenParams
from torch.utils.data import DataLoader
from utils.timer import Timer
from utils.loader_utils import FineSampler, get_stamp_list

from utils.image_utils import psnr
from lpipsPyTorch import lpips
from utils.loss_utils import l1_loss, ssim, l2_loss, lpips_loss
from pytorch_msssim import ms_ssim

from utils.scene_utils import render_training_image
from time import time
import copy
from gaussian_renderer import render, network_gui

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


to8b = lambda x : (255*np.clip(x.cpu().numpy(),0,1)).astype(np.uint8)

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

from train import scene_reconstruction

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

class GUI:
    def train_coarse(self):
        scene_reconstruction(
                self.dataset, 
                self.opt, 
                self.hyperparams, 
                self.pipe, 
                self.testing_iterations, 
                self.saving_iterations,
                self.checkpoint_iterations,
                self.checkpoint,
                self.debug_from,
                self.gaussians, 
                self.scene, 
                "coarse", 
                self.tb_writer, 
                self.opt.coarse_iterations,
                self.timer)
    
    def init_taining(self):
        first_iter = 1

        # Set up gaussian training
        self.gaussians.training_setup(self.opt)

        # Load from fine model if it exists
        if self.checkpoint:
            if 'fine' in self.checkpoint: 
                (model_params, first_iter) = torch.load(self.checkpoint)
                self.gaussians.restore(model_params, self.opt)

        # Set current iteration
        self.iteration = first_iter

        # Define background
        bg_color = [1, 1, 1] if self.dataset.white_background else [0, 0, 0]
        self.background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        # Events for counting duration of step
        self.iter_start = torch.cuda.Event(enable_timing = True)
        self.iter_end = torch.cuda.Event(enable_timing = True)
        
        # 
        viewpoint_stack = None
        
        # Define loss and psnr values for logging
        self.ema_loss_for_log = 0.0
        self.ema_psnr_for_log = 0.0

        # Define final iteration of training
        self.final_iter = self.opt.iterations

        # Get the test and training cameras (contain the dataset via `.dataset[#]`)
        # video_cams = self.scene.getVideoCameras()
        self.test_cams = self.scene.getTestCameras()
        self.train_cams = self.scene.getTrainCameras()

        # If the `viewpoint_stack` variable is tbd and the optimiser has not been loaded
        if not viewpoint_stack and not self.opt.dataloader:
            # dnerf's branch
            viewpoint_stack = [i for i in self.train_cams]
            self.temp_list = copy.deepcopy(viewpoint_stack)

        self.test_viewpoint_stack = [i for i in self.test_cams]
        self.test_temp_list = copy.deepcopy(self.test_viewpoint_stack)


        batch_size = self.opt.batch_size

        # If the data loader has been defined, stack training data for data loading
        if self.opt.dataloader:
            viewpoint_stack = self.scene.getTrainCameras()
            if self.opt.custom_sampler is not None:
                print('Custom sampler loaded')
                sampler = FineSampler(viewpoint_stack)
                viewpoint_stack_loader = DataLoader(viewpoint_stack, batch_size=batch_size,sampler=sampler,num_workers=16,collate_fn=list)
                random_loader = False
            else:
                viewpoint_stack_loader = DataLoader(viewpoint_stack, batch_size=batch_size,shuffle=True,num_workers=16,collate_fn=list)
                random_loader = True
            loader = iter(viewpoint_stack_loader)
            self.viewpoint_stack_loader = viewpoint_stack_loader

        else:
            print('Data loader not loaded')
            random_loader = None
            loader = None

        self.loader = loader
        self.random_loader = random_loader
        self.viewpoint_stack = viewpoint_stack

        self.load_in_memory = False 
        
    def __init__(self, 
                 args, 
                 hyperparams, 
                 dataset, 
                 opt, 
                 pipe, 
                 testing_iterations, 
                 saving_iterations,
                 ckpt_it,
                 ckpt_start,
                 debug_from,
                 expname
                 ):

        self.dataset = dataset
        self.hyperparams = hyperparams
        self.args = args
        self.opt = opt
        self.pipe = pipe
        self.testing_iterations = testing_iterations
        self.saving_iterations = saving_iterations

        self.checkpoint_iterations = ckpt_it
        self.checkpoint = ckpt_start
        self.expname = expname
        self.debug_from = debug_from
        self.rotator = None

        self.tb_writer = prepare_output_and_logger(expname)
        self.gaussians = GaussianModel(dataset.sh_degree, hyperparams, )
        dataset.model_path = args.model_path
        
        self.timer = Timer()
        self.scene = Scene(dataset, self.gaussians, load_coarse=None)
        self.timer.start()

        # Train the steps for the coarse field
        self.train_coarse()

        self.init_taining()

        try:
            self.W, self.H = self.scene.getTestCameras().dataset[0].image.shape[2], self.scene.getTestCameras().dataset[0].image.shape[1]
            self.fovy = self.scene.getTestCameras().dataset[0].FovY
        except:
            self.W, self.H = self.scene.getTestCameras()[0].image_width, self.scene.getTestCameras()[0].image_height
            self.fovy = self.scene.getTestCameras()[0].FoVy

        self.mode = "rgb"
        self.buffer_image = np.ones((self.W, self.H, 3), dtype=np.float32)
        self.time = 0.

        self.buffer = None
        self.coarse_stage = True


        self.ema_loss_for_log = 0.0
        self.ema_psnr_for_log = 0.0

        self.rot_hist = []


    def generate_rotator(self, display=False):
        # Generate a starting point for the plane rotation using PCA
        with torch.no_grad():
            # Get the positions of the 3D Gaussians and rotate them w.r.t some parameter
            xyz = self.gaussians.get_xyz

            # Compute EigenDecomp pre-requisits
            mean = torch.mean(xyz, dim=0)
            xyz_cent = xyz - mean

            # Find Covariance Matric
            cov_mat = torch.mm(xyz_cent.t(), xyz_cent) / (xyz_cent.size(0) - 1)
            # Get Eigen Values and Vecotrs
            eigenvalues, eigenvectors = torch.linalg.eig(cov_mat)

            # Get projection
            transformed_data_np = torch.mm(xyz_cent, eigenvectors.real).detach().cpu().numpy()

            origin = torch.zeros(3).detach().cpu().numpy()
            basis_vectors = torch.eye(3).detach().cpu().numpy()
            # Project the original basis vectors onto the new PCA axes
            transformed_vectors = eigenvectors.real.T.detach().cpu().numpy()

            if display:
                import plotly.graph_objects as go
                x = transformed_data_np[:, 0]
                y = transformed_data_np[:, 1]
                z = transformed_data_np[:, 2]

                x1 = xyz_cent.detach().cpu().numpy()[:, 0]
                y1 = xyz_cent.detach().cpu().numpy()[:, 1]
                z1 = xyz_cent.detach().cpu().numpy()[:, 2]

                # Prepare data for plotting the original and transformed axes
                fig = go.Figure(data=[go.Scatter3d(
                    x=x,
                    y=y,
                    z=z,
                    mode='markers',
                    marker=dict(
                        size=2,
                        color='red',  # Color by the z-axis values for some gradient
                        opacity=0.2
                    )
                ), go.Scatter3d(
                    x=x1,
                    y=y1,
                    z=z1,
                    mode='markers',
                    marker=dict(
                        size=2,
                        color='blue',  # Color by the z-axis values for some gradient
                        opacity=0.2
                    )
                )])

                # Add the original axes (in blue)
                rgb = ['red', 'green', 'blue']
                for i in range(3):
                    fig.add_trace(go.Scatter3d(
                        x=[origin[0], basis_vectors[i, 0]],
                        y=[origin[1], basis_vectors[i, 1]],
                        z=[origin[2], basis_vectors[i, 2]],
                        mode='lines+markers',
                        marker=dict(size=4, color='blue'),
                        line=dict(color=rgb[i], width=4),
                        name=f'Original Axis {i + 1}'
                    ))

                # Add the PCA axes (in red)
                for i in range(3):
                    fig.add_trace(go.Scatter3d(
                        x=[origin[0], transformed_vectors[i, 0]],
                        y=[origin[1], transformed_vectors[i, 1]],
                        z=[origin[2], transformed_vectors[i, 2]],
                        mode='lines+markers',
                        marker=dict(size=4, color='red'),
                        line=dict(color=rgb[i], width=4),
                        name=f'PCA Axis {i + 1}'
                    ))

                # Set the layout for the plot
                fig.update_layout(
                    scene=dict(
                        xaxis_title='X',
                        yaxis_title='Y',
                        zaxis_title='Z',
                        aspectmode='cube',
                    ),
                    title='Projection from Original 3D Axes to PCA Axes',
                    showlegend=True
                )

                # Show the figure
                fig.show()

        # Set the rotation learnable parameters
        self.rotator = eigenvectors.real.detach()
        self.gaussians._deformation.deformation_net.grid.reorient_grid.data = eigenvectors.real #torch.nn.Parameter(self.rotator, requires_grad=True)

        torch.save(self.rotator, os.path.join(self.scene.model_path, 'rotator.pth'))

        print('Re-Orienting Grid Params: ', self.gaussians._deformation.deformation_net.grid.reorient_grid)

    # gui mode
    def render(self):
        # Get rotation initialisation start point
        self.generate_rotator()

        with tqdm(total=self.final_iter) as pbar:
            while self.iteration <= self.final_iter:
                if self.iteration <= self.final_iter:
                    self.train_step()
                    self.train_step('LR')
                    self.iteration += 1
                    pbar.update(1)

                else:
                    exit()
                
                # if (self.iteration % 100) == 0:
                #    self.test_step()

    def render_train_step(self,viewpoint_cams, LR_flag=False):
        if LR_flag: stage='fine_LR'
        else: stage = 'fine'
        
        images = []
        gt_images = []
        radii_list = []
        visibility_filter_list = []
        viewspace_point_tensor_list = []
        for viewpoint_cam in viewpoint_cams:
            
            render_pkg = render(viewpoint_cam, self.gaussians, self.pipe, self.background, stage=stage, cam_type=self.scene.dataset_type)
            
            image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
            images.append(image.unsqueeze(0))
            
            if self.scene.dataset_type!="PanopticSports":
                gt_image = viewpoint_cam.original_image.cuda()
            else:
                gt_image  = viewpoint_cam['image'].cuda()
            
            gt_images.append(gt_image.unsqueeze(0))
            
            if LR_flag == False:
                radii_list.append(radii.unsqueeze(0))
                visibility_filter_list.append(visibility_filter.unsqueeze(0))
                viewspace_point_tensor_list.append(viewspace_point_tensor)
        
        if LR_flag == False:
            radii = torch.cat(radii_list, 0).max(dim=0).values
            visibility_filter = torch.cat(visibility_filter_list).any(dim=0)
        image_tensor = torch.cat(images,0)
        gt_image_tensor = torch.cat(gt_images,0)
                
        if LR_flag: return psnr(image_tensor, gt_image_tensor).mean().double() # l1_loss(LR_tensor, gt_image_tensor[:,:3,:,:])
        
        # Loss
        Ll1 = l1_loss(image_tensor, gt_image_tensor[:,:3,:,:])

        psnr_ = psnr(image_tensor, gt_image_tensor).mean().double()
	
        loss = Ll1
        
        if self.hyperparams.time_smoothness_weight != 0:
            tv_loss = self.gaussians.compute_regulation(self.hyperparams.time_smoothness_weight, self.hyperparams.l1_time_planes, self.hyperparams.plane_tv_weight)
            loss += tv_loss
        if self.opt.lambda_dssim != 0:
            ssim_loss = ssim(image_tensor,gt_image_tensor)
            loss += self.opt.lambda_dssim * (1.0-ssim_loss)
	
        return loss,psnr_,Ll1, viewspace_point_tensor_list, visibility_filter, radii, viewspace_point_tensor

    def train_step(self, LR_flag=False):

        # Start recording step duration
        self.iter_start.record()

        # Update Gaussian lr for current iteration
        self.gaussians.update_learning_rate(self.iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if self.iteration % 1000 == 0:
            self.gaussians.oneupSHdegree()

      
        # If data exists
        if self.opt.dataloader and not self.load_in_memory:
            try:
                viewpoint_cams = next(self.loader)
            except StopIteration:
                print("reset dataloader into random dataloader.")
                if not self.random_loader:
                    viewpoint_stack_loader = DataLoader(self.viewpoint_stack, batch_size=self.opt.batch_size,shuffle=True,num_workers=32,collate_fn=list)
                    self.random_loader = True
                else:
                    viewpoint_stack_loader = self.viewpoint_stack_loader
                self.loader = iter(viewpoint_stack_loader)
                viewpoint_cams = next(self.loader)
        else:
            idx = 0
            viewpoint_cams = []
            
            # Construct batch for current step
            while idx < self.opt.batch_size :        
                viewpoint_cam = self.viewpoint_stack.pop(randint(0,len(self.viewpoint_stack)-1))
                
                # If viewpoint stack doesn't exist get it from the temp view point list
                if not self.viewpoint_stack :
                    self.viewpoint_stack =  self.temp_list.copy()
                viewpoint_cams.append(viewpoint_cam)
                idx +=1

            
            # If there are not cameras to load then end the current iteration
            if len(viewpoint_cams) == 0:
                return None
        
        # Render
        if (self.iteration - 1) == self.debug_from:
            self.pipe.debug = True

        # Render and return preds
        loss,psnr_, Ll1,viewspace_point_tensor_list, visibility_filter, radii, viewspace_point_tensor = self.render_train_step(viewpoint_cams)
        torch.cuda.empty_cache()
        
        # If data exists
        if self.opt.dataloader and not self.load_in_memory:
            try:
                viewpoint_cams = next(self.loader)
            except StopIteration:
                print("reset dataloader into random dataloader.")
                if not self.random_loader:
                    viewpoint_stack_loader = DataLoader(self.viewpoint_stack, batch_size=self.opt.batch_size,shuffle=True,num_workers=32,collate_fn=list)
                    self.random_loader = True
                else:
                    viewpoint_stack_loader = self.viewpoint_stack_loader
                self.loader = iter(viewpoint_stack_loader)
                viewpoint_cams = next(self.loader)
        else:
            idx = 0
            viewpoint_cams = []
            
            # Construct batch for current step
            while idx < self.opt.batch_size :        
                viewpoint_cam = self.viewpoint_stack.pop(randint(0,len(self.viewpoint_stack)-1))
                
                # If viewpoint stack doesn't exist get it from the temp view point list
                if not self.viewpoint_stack :
                    self.viewpoint_stack =  self.temp_list.copy()
                viewpoint_cams.append(viewpoint_cam)
                idx +=1

            
            # If there are not cameras to load then end the current iteration
            if len(viewpoint_cams) == 0:
                return None
                
        loss += self.render_train_step(viewpoint_cams, LR_flag=True)
        loss.backward()

        # Error if loss becomes nan
        if torch.isnan(loss).any():
            print("loss is nan,end training, reexecv program now.")
            os.execv(sys.executable, [sys.executable] + sys.argv)
        
        viewspace_point_tensor_grad = torch.zeros_like(viewspace_point_tensor)
        for idx in range(0, len(viewspace_point_tensor_list)):
            viewspace_point_tensor_grad = viewspace_point_tensor_grad + viewspace_point_tensor_list[idx].grad
        
        # Record end of step
        self.iter_end.record()

        # Log and save
        with torch.no_grad():
            self.timer.pause()
            if (self.iteration % 10) == 0:
                self.ema_loss_for_log = 0.4 * loss.item() + 0.6 * self.ema_loss_for_log
                self.ema_psnr_for_log = 0.4 * psnr_ + 0.6 * self.ema_psnr_for_log
                total_point = self.gaussians._xyz.shape[0]
                
            training_report(self.tb_writer, self.iteration, Ll1, loss, l1_loss, self.iter_start.elapsed_time(self.iter_end), self.checkpoint_iterations, self.scene, render, [self.pipe, self.background], 'fine', self.scene.dataset_type)
            
            # Save scene when at the saving iteration
            if (self.iteration in self.saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(self.iteration))
                self.scene.save(self.iteration, 'fine')
            
            # Render images
            if self.dataset.render_process:
                if (self.iteration < 1000 and self.iteration % 10 == 9) \
                    or (self.iteration < 3000 and self.iteration % 50 == 49) \
                        or (self.iteration < 60000 and self.iteration %  100 == 99) :
                    # breakpoint()
                        render_training_image(self.scene, self.gaussians, [self.test_cams[self.iteration%len(self.test_cams)]], render, self.pipe, self.background, "finetest", self.iteration,self.timer.get_elapsed_time(),self.scene.dataset_type)
                        render_training_image(self.scene, self.gaussians, [self.train_cams[self.iteration%len(self.train_cams)]], render, self.pipe, self.background, "finetrain", self.iteration,self.timer.get_elapsed_time(),self.scene.dataset_type)
                        # render_training_image(scene, gaussians, train_cams, render, pipe, background, stage+"train", iteration,timer.get_elapsed_time(),scene.dataset_type)
            
            self.timer.start()
            
            # Densification
            if self.iteration < self.opt.densify_until_iter :
                # Keep track of max radii in image-space for pruning
                self.gaussians.max_radii2D[visibility_filter] = torch.max(self.gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                self.gaussians.add_densification_stats(viewspace_point_tensor_grad, visibility_filter)
   
                opacity_threshold = self.opt.opacity_threshold_fine_init - self.iteration*(self.opt.opacity_threshold_fine_init - self.opt.opacity_threshold_fine_after)/(self.opt.densify_until_iter)  
                densify_threshold = self.opt.densify_grad_threshold_fine_init - self.iteration*(self.opt.densify_grad_threshold_fine_init - self.opt.densify_grad_threshold_after)/(self.opt.densify_until_iter)  
                
                if  self.iteration > self.opt.densify_from_iter and self.iteration % self.opt.densification_interval == 0 and self.gaussians.get_xyz.shape[0]<300000: # 360000
                    size_threshold = 20 if self.iteration > self.opt.opacity_reset_interval else None
                    self.gaussians.densify(densify_threshold, opacity_threshold, self.scene.cameras_extent, size_threshold, 5, 5, self.scene.model_path, self.iteration, "first")
                
                if  self.iteration > self.opt.pruning_from_iter and self.iteration % self.opt.pruning_interval == 0 and self.gaussians.get_xyz.shape[0]>200000:# 200000
                    size_threshold = 20 if self.iteration > self.opt.opacity_reset_interval else None
                    self.gaussians.prune(densify_threshold, opacity_threshold, self.scene.cameras_extent, size_threshold)
                    
                # if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0 :
                if self.iteration % self.opt.densification_interval == 0 and self.gaussians.get_xyz.shape[0]<300000 and self.opt.add_point:#360000
                    self.gaussians.grow(5,5, self.scene.model_path, self.iteration, "fine")
                    # torch.cuda.empty_cache()
                
                if self.iteration % self.opt.opacity_reset_interval == 0:
                    self.gaussians.reset_opacity()

            # Optimizer step
            if self.iteration < self.opt.iterations:
                self.gaussians.optimizer.step()
                self.gaussians.optimizer.zero_grad(set_to_none = True)

            
            if (self.iteration in self.checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(self.iteration))
                torch.save((self.gaussians.capture(), self.iteration), self.scene.model_path + "/chkpnt" + f"_{'fine'}_" + str(self.iteration) + ".pth")
                torch.save(self.rotator, os.path.join(self.scene.model_path,'rotator.pth'))

    def update_rotation_hist(self, rot):
        if rot.size(0) == 1: rot= rot.squeeze(0)

        if self.rot_hist == []:
            self.rot_hist = rot.unsqueeze(-1)
        else:
            self.rot_hist = torch.cat((self.rot_hist, rot.unsqueeze(-1)), dim=-1)

    def plot_rotation_params(self, rot):

        self.update_rotation_hist(rot)

        plt.figure(figsize=(6, 4))
        if self.rot_hist[0,0].size(-1) % 5 == 0:
            x = [i for i in range(self.rot_hist[0,0].size(-1))]

            colors = np.zeros((3, 3, 3))  # Initialize RGB color array
            x_col = np.linspace(0, 1, 3)  # Normalized x-values from 0 to 1
            y_col = np.linspace(0, 1, 3)
            X, Y = np.meshgrid(x_col, y_col)
            colors[..., 0] = Y  # Red channel increases with Y
            colors[..., 2] = X

            for i in range(3):
                for j in range(3):
                    plt.plot(x, self.rot_hist[i][j].cpu().numpy(),
                             color=colors[i, j, :])
            plt.savefig('rotation_history.png')


    @torch.no_grad()
    def test_step(self):

        self.plot_rotation_params(self.gaussians._deformation.deformation_net.grid.reorient_grid)

        idx = 0
        if self.iteration < (self.final_iter -1):
            idx = 0
            viewpoint_cams = []
            
            # Construct batch for current step
            while idx < 10:        
                viewpoint_cam = self.test_viewpoint_stack.pop(randint(0,len(self.test_viewpoint_stack)-1))
                
                # If viewpoint stack doesn't exist get it from the temp view point list
                if not self.test_viewpoint_stack :
                    self.test_viewpoint_stack =  self.test_temp_list.copy()
                viewpoint_cams.append(viewpoint_cam)
                idx +=1
        else:
            viewpoint_cams = self.test_viewpoint_stack

        images = []
        gt_images = []
        radii_list = []
        visibility_filter_list = []
        viewspace_point_tensor_list = []

        # Render and return preds
        for viewpoint_cam in viewpoint_cams:
            render_pkg = render(viewpoint_cam, self.gaussians, self.pipe, self.background, stage='fine',cam_type=self.scene.dataset_type)
            image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
            
            images.append(image.unsqueeze(0))
            if self.scene.dataset_type!="PanopticSports":
                gt_image = viewpoint_cam.original_image.cuda()
            else:
                gt_image  = viewpoint_cam['image'].cuda()
            
            gt_images.append(gt_image.unsqueeze(0))
            radii_list.append(radii.unsqueeze(0))
            visibility_filter_list.append(visibility_filter.unsqueeze(0))
            viewspace_point_tensor_list.append(viewspace_point_tensor)
        

        radii = torch.cat(radii_list, 0).max(dim=0).values
        visibility_filter = torch.cat(visibility_filter_list).any(dim=0)
        
        # Loss
        # breakpoint()
        ssims = []
        ms_ssims = []
        dssims = []
        psnrs = []
        lpipss = []
        lpipsa = []

        # Only compute extra metrics at the end of training -> can be slow
        if self.iteration < (self.final_iter -1):
            for idx in range(len(gt_images)):
                    psnrs.append(psnr(images[idx], gt_images[idx]))
                    ssims.append(ssim(images[idx], gt_images[idx]))            
        else:
            for idx in range(len(gt_images)):
                psnrs.append(psnr(images[idx], gt_images[idx]))

                ssims.append(ssim(images[idx], gt_images[idx]))            
                ms_ssims.append(ms_ssim(images[idx], gt_images[idx],data_range=1, size_average=True ))
                dssims.append((1-ms_ssims[-1])/2)


                lpipss.append(lpips(images[idx], gt_images[idx], net_type='vgg'))
                lpipsa.append(lpips(images[idx], gt_images[idx], net_type='alex'))
              


def prepare_output_and_logger(expname):    
    if not args.model_path:
        # if os.getenv('OAR_JOB_ID'):
        #     unique_str=os.getenv('OAR_JOB_ID')
        # else:
        #     unique_str = str(uuid.uuid4())
        unique_str = expname

        args.model_path = os.path.join("./output/", unique_str)
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer
def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, stage, dataset_type):
    if tb_writer:
        tb_writer.add_scalar(f'{stage}/train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar(f'{stage}/train_loss_patchestotal_loss', loss.item(), iteration)
        tb_writer.add_scalar(f'{stage}/iter_time', elapsed, iteration)
        
    
    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        # 
        validation_configs = ({'name': 'test', 'cameras' : [scene.getTestCameras()[idx % len(scene.getTestCameras())] for idx in range(10, 5000, 299)]},
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(10, 5000, 299)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians,stage=stage, cam_type=dataset_type, *renderArgs)["render"], 0.0, 1.0)
                    if dataset_type == "PanopticSports":
                        gt_image = torch.clamp(viewpoint["image"].to("cuda"), 0.0, 1.0)
                    else:
                        gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    try:
                        if tb_writer and (idx < 5):
                            tb_writer.add_images(stage + "/"+config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                            if iteration == testing_iterations[0]:
                                tb_writer.add_images(stage + "/"+config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    except:
                        pass
                    l1_test += l1_loss(image, gt_image).mean().double()
                    # mask=viewpoint.mask
                    
                    psnr_test += psnr(image, gt_image, mask=None).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                # print("sh feature",scene.gaussians.get_features.shape)
                if tb_writer:
                    tb_writer.add_scalar(stage + "/"+config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(stage+"/"+config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram(f"{stage}/scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            
            tb_writer.add_scalar(f'{stage}/total_points', scene.gaussians.get_xyz.shape[0], iteration)
            tb_writer.add_scalar(f'{stage}/deformation_rate', scene.gaussians._deformation_table.sum()/scene.gaussians.get_xyz.shape[0], iteration)
            tb_writer.add_histogram(f"{stage}/scene/motion_histogram", scene.gaussians._deformation_accum.mean(dim=-1)/100, iteration,max_bins=500)
        
        torch.cuda.empty_cache()
def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

if __name__ == "__main__":
    # Set up command line argument parser
    torch.cuda.empty_cache()
    parser = ArgumentParser(description="Training script parameters")
    setup_seed(6666)
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    hp = ModelHiddenParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[3000,7000,14000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[14000,20000, 30000, 45000, 60000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--expname", type=str, default = "")
    parser.add_argument("--configs", type=str, default = "")
    parser.add_argument('--gui', action='store_true', default=False)

    
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    if args.configs:
        import mmcv
        from utils.params_utils import merge_hparams
        config = mmcv.Config.fromfile(args.configs)
        args = merge_hparams(args, config)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    torch.autograd.set_detect_anomaly(args.detect_anomaly)


    gui = GUI(args=args, hyperparams=hp.extract(args), dataset=lp.extract(args), opt=op.extract(args), pipe=pp.extract(args),testing_iterations=args.test_iterations, saving_iterations=args.save_iterations,
                ckpt_it=args.checkpoint_iterations, ckpt_start=args.start_checkpoint, debug_from=args.debug_from, expname=args.expname)

    gui.render()

    # training( args.checkpoint_iterations, args.start_checkpoint, args.debug_from, args.expname)
    # All done
    print("\nTraining complete.")
