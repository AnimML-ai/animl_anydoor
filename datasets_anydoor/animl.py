import json
import cv2
import numpy as np
import os, glob

from numpy.ma.core import concatenate
from torch.utils.data import Dataset
from PIL import Image
import cv2
from .data_utils import * 
from .base import BaseDataset
import albumentations as A

def pad_to_square(image, pad_value = 255, random = False):
    H,W = image.shape[0], image.shape[1]
    if H == W:
        return image

    padd = abs(H - W)
    if random:
        padd_1 = int(np.random.randint(0,padd))
    else:
        padd_1 = int(padd / 2)
    padd_2 = padd - padd_1

    if H > W:
        pad_param = ((0,0),(padd_1,padd_2),(0,0)) if len(image.shape) == 3 else ((0,0),(padd_1,padd_2))
    else:
        pad_param = ((padd_1,padd_2),(0,0),(0,0)) if len(image.shape) == 3 else ((padd_1,padd_2),(0,0))

    image = np.pad(image, pad_param, 'constant', constant_values=pad_value)
    return image

def expand_image(image, ratio, expand_value):
    h, w = image.shape[0], image.shape[1]
    H, W = int(h * ratio), int(w * ratio)
    h1 = int((H - h) // 2)
    h2 = H - h - h1
    w1 = int((W - w) // 2)
    w2 = W - w - w1

    pad_param_image = ((h1, h2), (w1, w2), (0, 0)) if len(image.shape) == 3 else ((h1, h2), (w1, w2))
    image = np.pad(image, pad_param_image, 'constant', constant_values=expand_value)
    return image


class AnimlDataset(BaseDataset):
    def __init__(self, data_dir):
        obj_list = glob.glob(f'{data_dir}/*/**/pair_mask_train_data')
        self.data = []
        for obj in obj_list:
            img_list = sorted(glob.glob(f'{obj}/render_data/*'))
            mask_list = sorted(glob.glob(f'{obj}/mask_data/*'))
            self.data += zip(img_list, mask_list)

        self.size = (512,512)
        self.clip_size = (224,224)
        self.dynamic = 2

    def __len__(self):
        return len(self.data)

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

        image_path, mask_path = self.data[idx]

        # Read Image and Mask
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (image.shape[1]//2, image.shape[0]//2))

        tar_image = cv2.rotate(image[:image.shape[0]//2], cv2.ROTATE_90_CLOCKWISE)
        ref_image = cv2.rotate(image[image.shape[0]//2:], cv2.ROTATE_90_CLOCKWISE)

        mask = Image.open(mask_path).convert('P')
        mask = np.array(mask)
        mask = cv2.resize(mask, (mask.shape[1]//2, mask.shape[0]//2))

        tar_mask = cv2.rotate((mask[:mask.shape[0]//2] > 128).astype(np.uint8), cv2.ROTATE_90_CLOCKWISE)
        ref_mask = cv2.rotate((mask[mask.shape[0]//2:] > 128).astype(np.uint8), cv2.ROTATE_90_CLOCKWISE)

        item_with_collage = self.process_pairs(ref_image, ref_mask, tar_image, tar_mask, max_ratio = 1.0)
        sampled_time_steps = self.sample_timestep()
        item_with_collage['time_steps'] = sampled_time_steps
        return item_with_collage

    def process_pairs(self, ref_image, ref_mask, tar_image, tar_mask, max_ratio=0.8):
        assert mask_score(ref_mask) > 0.90
        assert self.check_mask_area(ref_mask) == True
        assert self.check_mask_area(tar_mask) == True

        ### REFERENCE ###
        ref_box_yyxx = get_bbox_from_mask(ref_mask)
        assert self.check_region_size(ref_mask, ref_box_yyxx, ratio=0.10, mode='min') == True

        # Filtering background for the reference image
        ref_mask_3 = np.stack([ref_mask, ref_mask, ref_mask], -1)
        masked_ref_image = ref_image * ref_mask_3 + np.ones_like(ref_image) * 255 * (1 - ref_mask_3)
        ref_image_collage = sobel(masked_ref_image, ref_mask)

        obj_y1, obj_y2, obj_x1, obj_x2 = ref_box_yyxx
        obj_center_y, obj_center_x = (obj_y1 + obj_y2) // 2, (obj_x1 + obj_x2) // 2

        dilated_obj_ratio = np.random.randint(11, 15) / 10
        collage = ref_image_collage[obj_y1:obj_y2, obj_x1:obj_x2]
        collage = expand_image(collage, dilated_obj_ratio, 0)
        collage = pad_to_square(collage, pad_value=0)
        H2, W2 = collage.shape[0], collage.shape[1]

        dilated_square_obj_delta = collage.shape[0] // 2
        dil_ref_bb_y1, dil_ref_bb_x1 = obj_center_y - dilated_square_obj_delta, obj_center_x - dilated_square_obj_delta
        dil_ref_bb_y2, dil_ref_bb_x2 = dil_ref_bb_y1 + collage.shape[0], dil_ref_bb_x1 + collage.shape[1]

        collage_mask = ref_mask[obj_y1:obj_y2, obj_x1:obj_x2].astype(np.float32)
        collage_mask = expand_image(collage_mask, dilated_obj_ratio, 0)
        collage_mask = pad_to_square(collage_mask, pad_value=2)

        #### TARGET ###
        # cropped_target_image = tar_image.copy()
        # cropped_target_image[dil_ref_bb_y1:dil_ref_bb_y2, dil_ref_bb_x1:dil_ref_bb_x2] = np.where(tar_mask[dil_ref_bb_y1:dil_ref_bb_y2, dil_ref_bb_x1:dil_ref_bb_x2, None] > 0, collage, cropped_target_image[dil_ref_bb_y1:dil_ref_bb_y2, dil_ref_bb_x1:dil_ref_bb_x2])

        hint = tar_image.copy()
        hint[dil_ref_bb_y1:dil_ref_bb_y2, dil_ref_bb_x1:dil_ref_bb_x2] = np.where(tar_mask[dil_ref_bb_y1:dil_ref_bb_y2, dil_ref_bb_x1:dil_ref_bb_x2, None] > 0, collage, hint[dil_ref_bb_y1:dil_ref_bb_y2, dil_ref_bb_x1:dil_ref_bb_x2])
        hint_mask = tar_mask.copy()
        hint_mask[dil_ref_bb_y1:dil_ref_bb_y2, dil_ref_bb_x1:dil_ref_bb_x2] = np.where(tar_mask[dil_ref_bb_y1:dil_ref_bb_y2, dil_ref_bb_x1:dil_ref_bb_x2] > 0, collage_mask, np.full_like(collage_mask, 2))

        tar_box_yyxx = get_bbox_from_mask(tar_mask)
        tar_box_yyxx = expand_bbox(tar_mask, tar_box_yyxx, ratio=[1.1,1.2]) #1.1  1.3
        assert self.check_region_size(tar_mask, tar_box_yyxx, ratio = max_ratio, mode = 'max') == True

        tar_box_yyxx_crop = expand_bbox(tar_image, tar_box_yyxx, ratio=[1.3, 3.0])
        tar_box_yyxx_crop = box2squre(tar_image, tar_box_yyxx_crop) # crop box
        y1,y2,x1,x2 = tar_box_yyxx_crop

        cropped_target_image = tar_image[y1:y2,x1:x2,:]
        cropped_tar_mask = tar_mask[y1:y2,x1:x2]

        collage = hint[y1:y2,x1:x2]
        collage_mask = hint_mask[y1:y2,x1:x2]

        H1, W1 = cropped_target_image.shape[0], cropped_target_image.shape[1]
        H2, W2 = cropped_target_image.shape[0], cropped_target_image.shape[1]

        ### DEBUG ###
        cv2.imwrite('/tmp/masked_ref_image.jpg', cv2.cvtColor(masked_ref_image, cv2.COLOR_BGR2RGB))
        cv2.imwrite('/tmp/cropped_target_image.jpg', cv2.cvtColor(cropped_target_image, cv2.COLOR_BGR2RGB))
        cv2.imwrite('/tmp/collage.png', cv2.cvtColor(np.concatenate([collage, collage_mask.astype(np.uint8)[..., None]], axis=-1), cv2.COLOR_BGR2RGB))

        ### RESIZING ###
        masked_ref_image = masked_ref_image[dil_ref_bb_y1:dil_ref_bb_y2, dil_ref_bb_x1:dil_ref_bb_x2]
        masked_ref_image = cv2.resize(masked_ref_image.astype(np.uint8), (224,224) ).astype(np.uint8)

        cropped_target_image = cv2.resize(cropped_target_image.astype(np.uint8), (512, 512)).astype(np.float32)

        collage = cv2.resize(collage.astype(np.uint8), (512, 512)).astype(np.float32)
        collage_mask = cv2.resize(collage_mask.astype(np.uint8), (512, 512), interpolation=cv2.INTER_NEAREST).astype(np.float32)
        collage_mask[collage_mask == 2] = -1

        ### NORMALISATION ###
        masked_ref_image = masked_ref_image / 255
        cropped_target_image = cropped_target_image / 127.5 - 1.0
        collage = collage / 127.5 - 1.0
        collage = np.concatenate([collage, collage_mask[..., None]], -1)

        item = dict(
            ref=masked_ref_image.copy(),
            jpg=cropped_target_image.copy(),
            hint=collage.copy(),
            extra_sizes=np.array([H1, W1, H2, W2]),
            tar_box_yyxx_crop=np.array(tar_box_yyxx_crop)
        )
        return item
