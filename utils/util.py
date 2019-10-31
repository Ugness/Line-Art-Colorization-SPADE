import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt
import os

def resize_img(img, size=768, pad_value=255):
    # (H, W, C)
    H, W, C = img.shape
    dtype = img.dtype
    canvas = np.ones((size, size, C), dtype=dtype) * pad_value
    if H > W:
        new_H = size    # size / H * H
        new_W = int(size / H * W)
        img = cv2.resize(img, (new_W, new_H), interpolation=cv2.INTER_LINEAR)
        start = int((size - new_W) // 2)
        canvas[:, start:start+new_W, :] = img
    else:
        new_W = size
        new_H = int(size / W * H)
        img = cv2.resize(img, (new_W, new_H), interpolation=cv2.INTER_LINEAR)
        start = int((size - new_H) // 2)
        canvas[start:start+new_H, :, :] = img
    return canvas


if __name__ == '__main__':
    data_dir = ['data', 'safebooru', 'upper_body', 'color']
    data_dir = os.path.join(*data_dir)
    img_path = os.listdir(data_dir)[1]
    img = cv2.imread(os.path.join(data_dir, img_path))
    print(img.shape)
    img = resize_img(img)
    # plt.figure()
    # plt.imshow(img)
    # plt.show()
    cv2.imwrite('sample.jpg', img)
