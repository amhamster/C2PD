# -*- coding: utf-8 -*-

import os
import math
import time
import torch
import shutil
import random
import numpy as np
from PIL import Image
import torch.optim as optim
from collections import Iterable
from skimage.transform import resize


def mod_crop(img, modulo):
    if len(img.shape) == 2:
        h, w = img.shape
        return img[: h - (h % modulo), :w - (w % modulo)]
    else:
        _, h, w = img.shape
        return img[:, : h - (h % modulo), :w - (w % modulo)]


def get_patch(*args, patch_size=64, scale=4):
    ih, iw = args[0].shape[1:]
    tp = scale * patch_size
    ip = tp // scale

    iy = random.randrange(0, ih - ip + 1)
    ix = random.randrange(0, iw - ip + 1)
    tx, ty = scale * ix, scale * iy
    ret = [
        args[0][:, iy:iy + ip, ix:ix + ip],
        *[a[:, ty:ty + tp, tx:tx + tp] for a in args[1:]]
    ]

    return ret


def augment(*args, hflip=True, rot=True):
    hflip = hflip and random.random() < 0.5
    vflip = rot and random.random() < 0.5
    rot90 = random.random() < 0.5

    def _augment(img):
        if hflip: img = img[:, :, ::-1]
        if vflip: img = img[:, ::-1, :]
        if rot90: img = img.transpose(0, 2, 1)
        return np.ascontiguousarray(img)

    return [_augment(a) for a in args]


def get_lowers(im_np, factor):
    if im_np.ndim == 3:
        im_np = im_np.transpose(1, 2, 0)

    h0, w0 = im_np.shape[:2]
    h, w = int(math.ceil(h0 / float(factor))), int(math.ceil(w0 / float(factor)))

    if h0 != h * factor or w0 != w * factor:
        im_np = resize(im_np, (h * factor, w * factor), order=1, mode='reflect', clip=False, preserve_range=True,
                       anti_aliasing=True)

    if len(im_np.shape) == 3:
        im_np = im_np[:, :, 0]
        lowers = np.expand_dims(np.array(Image.fromarray(im_np).resize((w, h), Image.BICUBIC)), 2)
    else:
        lowers = np.array(Image.fromarray(im_np).resize((w, h), Image.BICUBIC))

    if lowers.ndim == 3:
        lowers = lowers.transpose((2, 0, 1))

    return lowers


def bic_resize(im_np, h=256, w=256):

    if im_np.ndim == 3:
        im_np = im_np.transpose(1, 2, 0)

    if len(im_np.shape) == 3:
            im_np = im_np[:, :, 0]
            lowers = np.expand_dims(np.array(Image.fromarray(im_np).resize((w, h), Image.BICUBIC)), 2)
    else:
        lowers = np.array(Image.fromarray(im_np).resize((w, h),Image.BICUBIC))

    if lowers.ndim == 3:
        lowers = lowers.transpose((2, 0, 1))

    return lowers


def np_to_tensor(*args, input_data_range=1.0, process_data_range=1.0):
    def _np_to_tensor(img):
        np_transpose = img.astype(np.float32)
        tensor = torch.from_numpy(np_transpose).float()
        tensor.mul_(process_data_range / input_data_range)
        return tensor.float()

    return [_np_to_tensor(a) for a in args]


def root_mean_sqrt_error(im_pred, im_true, min_value, max_value, border=6, mul_ratio=100, is_train=False):
    b, c, h, w = im_true.size()
    if not is_train:
        im_pred, im_true = im_pred.reshape(im_pred.size(0), -1), im_true.reshape(im_true.size(0), -1)
        im_pred = im_pred * (max_value - min_value) + min_value
    if border != 0:
        im_pred = im_pred.view(b, c, h, w)[:, :, border: -border, border: -border]
        im_true = im_true.view(b, c, h, w)[:, :, border: -border, border: -border]

    return round(torch.sqrt(torch.mean(((im_true * mul_ratio) - (im_pred * mul_ratio)) ** 2)).item(), 5), im_pred #rmse


def make_checkpoint_dir(file_name):
    path = './checkpoints/{}'.format(file_name)
    if os.path.isdir(path):
        shutil.rmtree(path)

    os.makedirs(path, exist_ok=True)


def init_state():
    np.random.seed(60)
    torch.manual_seed(60)
    torch.cuda.manual_seed(60)
    torch.cuda.manual_seed_all(60)
    torch.backends.cudnn.deterministic = True


def time_since(since):
    s = time.time() - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def make_optimizer(args, targets):

    return optim.Adam(targets.parameters(), lr=args.lr, weight_decay=args.weight_decay)
