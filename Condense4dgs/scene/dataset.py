from torch.utils.data import Dataset
from scene.cameras import Camera
import torch
from utils.graphics_utils import focal2fov


class FourDGSdataset(Dataset):
    def __init__(
        self,
        dataset,
        args,
        dataset_type
    ):
        self.dataset = dataset
        self.args = args
        self.dataset_type=dataset_type

    def __getitem__(self, index):
        if self.dataset_type != "PanopticSports":
            if self.dataset_type == 'condense':
                image, w2c, time = self.dataset[index]
                R, T = w2c
                FovX, FovY = self.dataset.load_fov(index)

                if image.shape[0] == 4:
                    mask = image[3]
                    image = image[:3]
                else: mask = None

                rgb_cam = Camera(
                    colmap_id=index, R=R, T=T, FoVx=FovX, FoVy=FovY,image=image, gt_alpha_mask=mask,mask=mask,
                    image_name=f"{index}", uid=index, data_device=torch.device("cuda"), time=time
                )

                return rgb_cam , self.dataset.get_pcd_path(index)

            else:
                try:
                    image, w2c, time = self.dataset[index]
                    R, T = w2c
                    FovX = focal2fov(self.dataset.focal[0], image.shape[2])
                    FovY = focal2fov(self.dataset.focal[0], image.shape[1])
                    mask = None
                except:
                    caminfo = self.dataset[index]
                    image = caminfo.image
                    R = caminfo.R
                    T = caminfo.T
                    FovX = caminfo.FovX
                    FovY = caminfo.FovY
                    time = caminfo.time

                    mask = caminfo.mask

            return Camera(colmap_id=index, R=R, T=T, FoVx=FovX, FoVy=FovY, image=image, gt_alpha_mask=mask,
                          image_name=f"{index}", uid=index, data_device=torch.device("cuda"), time=time
                      )
        else:
            return self.dataset[index]


    def __len__(self):
        
        return len(self.dataset)
