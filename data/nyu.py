# -*- coding: utf-8 -*-

import os
import math
import glob
import utility
import random
import numpy as np
from torch.utils import data

random.seed(60)

data_dict = {
    'train': 'dataset/nyu_v2/gt/',
    'Lu': 'test_data/Lu/gt/',
    'test': 'test_data/NYU/gt/',
    'Middlebury': 'test_data/Middlebury/gt/',
    'RGBDD': 'test_data/RGBDD/gt/',
}

class NYU(data.Dataset):
    def __init__(self, args, attr='train'):
        self.first = 0
        self.args = args
        self.attr = attr
        self.module = 16
        self.real = args.real

        self.gt_imgs = []
        self.lr_imgs = []
        self.rgb_imgs = []

        self.img_list = sorted(glob.glob(data_dict['{}'.format(attr)] + '*.npy'))

        if self.attr == 'train':
            for img_name in self.img_list:
                self.gt_imgs.append(np.expand_dims(np.load(img_name), 0))
                tmp_rgb_img = np.load(img_name.replace('gt', 'rgb'))
                tmp_rgb_img = np.transpose(tmp_rgb_img, (2, 0, 1))
                self.rgb_imgs.append(np.float32(tmp_rgb_img) / 255.)
                
                if self.real:
                    self.lr_imgs.append(np.expand_dims(np.load(img_name.replace('gt', 'lr')), 0))

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, item):
        item = item % len(self.img_list)
        min_value = 0
        max_value = 1
        if self.attr == 'train':
            if self.real:
                lr_img, gt_img, rgb_img = utility.mod_crop(self.lr_imgs[item], modulo=self.module), utility.mod_crop(self.gt_imgs[item], modulo=self.module), utility.mod_crop(
                self.rgb_imgs[item], modulo=self.module)
                _,h,w = gt_img.shape
                s = self.args.scale
                lr_img = utility.bic_resize(lr_img, h//s,w//s)
                lr_img, gt_img, rgb_img = utility.get_patch(lr_img, gt_img, rgb_img, patch_size=self.args.patch_size//self.args.scale, scale=self.args.scale)
                if self.args.data_augment:
                    lr_img, gt_img, rgb_img = utility.augment(lr_img, gt_img, rgb_img)
            else:
                gt_img, rgb_img = utility.mod_crop(self.gt_imgs[item], modulo=self.module), utility.mod_crop(
                self.rgb_imgs[item], modulo=self.module)

                gt_img, rgb_img = utility.get_patch(gt_img, rgb_img, patch_size=self.args.patch_size, scale=1)
                if self.args.data_augment:
                    gt_img, rgb_img = utility.augment(gt_img, rgb_img)
    
                lr_img = utility.get_lowers(gt_img, factor=self.args.scale)
            
        else:
            gt_name = self.img_list[item]
            rgb_name = gt_name.replace('gt', 'rgb')
            gt_img, rgb_img = np.load(gt_name), np.load(rgb_name)
            module = max(self.args.scale, self.module)

            if self.real:
                lr_img = np.load(gt_name.replace('gt', 'lr'))
                h,w = gt_img.shape
                s = self.args.scale
                lr_img = utility.bic_resize(lr_img, h//s,w//s)
                min_value = np.min(lr_img)
                max_value = np.max(lr_img)
                lr_img = (lr_img - min_value) / (max_value - min_value)
            else:
                tmp_gt = utility.mod_crop(gt_img, modulo=module)
                min_value = np.min(tmp_gt)
                max_value = np.max(tmp_gt)
                tmp_gt = (tmp_gt - min_value) / (max_value - min_value)
                tmp_gt = (tmp_gt - np.min(tmp_gt)) / (np.max(tmp_gt) - np.min(tmp_gt))
                lr_img = utility.get_lowers(tmp_gt, factor=self.args.scale)

            lr_img, gt_img = np.expand_dims(lr_img, 0), np.expand_dims(gt_img, 0)
            rgb_img = np.float32(np.transpose(rgb_img, axes=(2, 0, 1))) / 255.

            gt_img, rgb_img = utility.mod_crop(gt_img, modulo=module), utility.mod_crop(rgb_img, modulo=module)

        lr_img, gt_img, rgb_img = utility.np_to_tensor(lr_img, gt_img, rgb_img)

        sample = {'lr_img': lr_img, 'gt_img': gt_img, 'rgb_img': rgb_img, 'min_value': min_value, 'max_value': max_value}
        return sample
