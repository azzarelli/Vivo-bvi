from torch.utils.data import Dataset
from thirdparty.gaussian_splatting.scene.cameras import Camera
import torch
from thirdparty.gaussian_splatting.utils.graphics_utils import focal2fov
from tqdm import tqdm

class FourDGSdataset(Dataset):
    def __init__(
            self,
            dataset,
            args,
            dataset_type
    ):
        self.args = args
        self.dataset_type = dataset_type
        # self.raydict = None

        # Dataset will be a list of CamInfos class (IO skeleton of Camera class)
        # self.dataset = dataset

        # We need to generate the camera obj/ we need a class that can generate the raydict
        # simultaneously, because we want to avoid charging the GPU with all the images all at once
        # we need a function that loads all non-image data
        # Thats fine, we can load the images into the GPU on the fly
        # We also need to construct the raydict (not quite sure why when its already done withing the Camera class)
        # Finally we want to access the datasets w.r.t time values so we will create a function that
        # stores pointers to list w.r.t the time variable

        with torch.no_grad():
            data = []
            raydict = {}
            timedict = {}
            for idx, item in tqdm(enumerate(dataset)):
                # if idx > 100:break
                name = f"{item.image_path.split('/')[-3]}"
                time = item.timestamp
                data.append(
                    Camera(
                        colmap_id=item.uid, uid=item.uid,
                        R=item.R, T=item.T, timestamp=item.timestamp,
                        FoVx=item.FovX, FoVy=item.FovY,
                        width=item.width, height=item.height,
                        image=item.image_path, image_name=name,

                        rayd=1 # Flag to generate ray infos
                ))

                time = round(time * int(item.image_name))
                if time not in timedict:
                    timedict[time] = []

                timedict[time].append(idx)

                if name not in raydict:
                    raydict[name] = torch.cat([data[-1].rayo, data[-1].rayd], dim=1).cuda()

                # We reset this/clear its mem allocation so we dont block up our GPU or CPU - instead we will reload it on the
                # fly
                data[-1].rayd = None
                data[-1].rayo = None

            self.dataset = data
            self.timedict = timedict
            self.raydict = raydict

    def get_dataset_item(self, index):
        # We dont want to compute the gradient calculations on these...
        with torch.no_grad():
            self.dataset[index].rays = self.raydict[self.dataset[index].image_name]
            # self.dataset[index].rayo = self.raydict[self.dataset[index].image_name][0, :3].unsqueeze(0)
            # self.dataset[index].rayd = self.raydict[self.dataset[index].image_name][0, 3:].unsqueeze(0)
            return self.dataset[index]

    def __len__(self):

        return len(self.dataset)
