import numpy as np
import cv2


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
