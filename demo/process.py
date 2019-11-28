import numpy as np
import cv2
from PIL import Image
import torchvision.transforms.functional as F

import sys

sys.path.append('..')
from utils.sketchKeras.helper import *


def sketch(mod, img):
    """
    :param mod: sketchkeras model
    :param img: img
    :return:
    """
    from_mat = img
    H, W, C = img.shape
    from_mat = from_mat.transpose((2, 0, 1))  # HWC -> CHW
    light_map = np.zeros(from_mat.shape, dtype=np.float)
    for channel in range(3):
        light_map[channel] = get_light_map_single(from_mat[channel])
    light_map = normalize_pic(light_map)
    light_map = resize_img_768_3d(light_map)
    line_mat = mod.predict(light_map, batch_size=1)
    line_mat = line_mat.transpose((3, 1, 2, 0))[0]
    line_mat = line_mat[0:H, 0:W, :]
    line_mat = np.amax(line_mat, 2)
    sketch_img = get_denoised_img(line_mat)
    return sketch_img


def resize_img(img, size=768, pad_value=255, mode='nearest'):
    # (H, W, C)
    H, W, C = img.shape
    dtype = img.dtype
    canvas = np.ones((size, size, C), dtype=dtype) * pad_value
    if mode == 'nearest':
        interpolation = cv2.INTER_NEAREST
    elif mode == 'bilinear':
        interpolation = cv2.INTER_LINEAR
    elif mode == 'bicubic':
        interpolation = cv2.INTER_CUBIC
    else:
        print("Select Mode nearest/bilinear")
        assert False
    if H > W:
        new_H = size  # size / H * H
        new_W = int(size / H * W)
        img = cv2.resize(img, (new_W, new_H), interpolation)
        if C == 1:
            img = np.expand_dims(img, axis=2)
        start = int((size - new_W) // 2)
        canvas[:, start:start + new_W, :] = img
    else:
        new_W = size
        new_H = int(size / W * H)
        img = cv2.resize(img, (new_W, new_H), interpolation)
        if C == 1:
            img = np.expand_dims(img, axis=2)
        start = int((size - new_H) // 2)
        canvas[start:start + new_H, :, :] = img
    return canvas


def normalize(image):
    return (image - 0.5) / 0.5


def denormalize(image):
    return (image * 0.5) + 0.5


def getitem(line, hint, refer=None):
    # Assuming hint_img / line_img already re-sized.

    hint_img = np.array(hint)
    line_img = np.array(line)

    mask_img = hint_img[:, :, [3, ]]
    hint_img = hint_img[:, :, [0, 1, 2, ]]

    # Target tensor: normalized Lab color image
    color_hint = np.concatenate([mask_img, hint_img], axis=2)
    hint_tensor = F.to_tensor(color_hint)
    line_img = F.to_tensor(line_img)
    hint_img[1:, :, :] = normalize(hint_img[1:, :, :])
    line_img = normalize(line_img)
    if refer is not None:
        refer = F.to_tensor(refer)
        refer = normalize(refer).unsqueeze(0).float()

    # Fit to SPADE
    real_image = refer
    sketch_tensor = line_img
    image_path = None

    return {'label': real_image,
            'instance': hint_tensor.unsqueeze(0).float(),
            'image': sketch_tensor.unsqueeze(0).float(),
            'path': image_path,
            }
