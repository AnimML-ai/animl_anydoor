import json
import cv2
import numpy as np
import os, glob
from torch.utils.data import Dataset
from PIL import Image
import cv2
from .data_utils import * 
from .base import BaseDataset
import albumentations as A

class AnimlDataset(BaseDataset):
    def __init__(self, data_dir):
        self.data = glob.glob(f'{data_dir}/*/**/pair_mask_train_data')
        self.size = (512,512)
        self.clip_size = (224,224)
        self.dynamic = 2

    def __len__(self):
        return len(self.data)*1

    def check_region_size(self, image, yyxx, ratio, mode = 'max'):
        pass_flag = True
        H,W = image.shape[0], image.shape[1]
        H,W = H * ratio, W * ratio
        y1,y2,x1,x2 = yyxx
        h,w = y2-y1,x2-x1
        if mode == 'max':
            if h > H and w > W:
                pass_flag = False
        elif mode == 'min':
            if h < H and w < W:
                pass_flag = False
        return pass_flag
            
    def get_sample(self, idx):

        image_path = glob.glob(f'{self.data[idx]}/render_data')
        mask_path = glob.glob(f'{self.data[idx]}/mask_data')
        ind_img = np.random.randint(len(image_path))

        image_path = image_path[ind_img]
        mask_path = mask_path[ind_img]

        # Read Image and Mask
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (image.shape[1]//2, image.shape[0]//2))

        ref_image = cv2.rotate(image[:image.shape[0]//2], cv2.ROTATE_90_CLOCKWISE)
        tar_image = cv2.rotate(image[image.shape[0]//2:], cv2.ROTATE_90_CLOCKWISE)

        mask = Image.open(mask_path).convert('P')
        mask = np.array(mask)
        mask = cv2.resize(mask, (mask.shape[1]//2, mask.shape[0]//2))

        ref_mask = cv2.rotate((mask[:mask.shape[0]//2] > 128).astype(np.uint8), cv2.ROTATE_90_CLOCKWISE)
        tar_mask = cv2.rotate((mask[mask.shape[0]//2:] > 128).astype(np.uint8), cv2.ROTATE_90_CLOCKWISE)

        print(ref_image.shape, ref_mask.shape, tar_image.shape, tar_mask.shape)

        item_with_collage = self.process_pairs(ref_image, ref_mask, tar_image, tar_mask, max_ratio = 1.0)
        sampled_time_steps = self.sample_timestep()
        item_with_collage['time_steps'] = sampled_time_steps
        return item_with_collage

