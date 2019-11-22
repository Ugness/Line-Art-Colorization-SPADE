import numpy as np
import cv2
from PIL import Image
import torchvision.transforms.functional as F


def resize_img(img, size=768, pad_value=255, mod='nearest'):
    # (H, W, C)
    H, W, C = img.shape
    dtype = img.dtype
    canvas = np.ones((size, size, C), dtype=dtype) * pad_value
    if mod == 'nearest':
        interpolation = cv2.INTER_NEAREST
    elif mod == 'bilinear':
        interpolation = cv2.INTER_LINEAR
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


def getitem(line, hint):
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

    # Fit to SPADE
    real_image = None
    sketch_tensor = line_img
    image_path = None

    return {'label': real_image,
            'instance': hint_tensor.unsqueeze(0).float(),
            'image': sketch_tensor.unsqueeze(0).float(),
            'path': image_path,
            }
