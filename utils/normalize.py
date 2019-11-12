import os.path
from argparse import ArgumentParser

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image

import utils.util as util
from dataloader.base_dataset import get_transform
from dataloader.image_folder import make_dataset


def get_normalize_value(opt):
    from tqdm import tqdm
    opt = opt
    root = opt.dataroot

    color_dir = os.path.join(root, 'color/')
    color_paths = make_dataset(color_dir)
    color_paths = sorted(color_paths)

    transform = transforms.ToTensor()

    mean_ = []
    min_ = []
    max_ = []

    for path in tqdm(color_paths):
        color_img = Image.open(path).convert('RGB')
        color_img = transform(color_img).unsqueeze(0)
        numpy_img = util.rgb2lab(color_img, opt)[0, :, :, :].numpy()

        img_mean = np.mean(numpy_img, axis=(1, 2))
        img_min = np.min(numpy_img, axis=(1, 2))
        img_max = np.max(numpy_img, axis=(1, 2))

        mean_.append(img_mean)
        min_.append(img_min)
        max_.append(img_max)

    mean_ = np.array(mean_).mean(axis=0)
    min_ = np.array(min_).min(axis=0)
    max_ = np.array(max_).max(axis=0)

    var_ = []
    for path in tqdm(color_paths):
        color_img = Image.open(path).convert('RGB')
        color_img = transform(color_img).unsqueeze(0)
        numpy_img = util.rgb2lab(color_img, opt)[0, :, :, :].numpy()
        # Should divide by N-1 for sample var, but just use N. N is large enough.
        img_var = np.mean(numpy_img-mean_, axis=(1, 2))
        var_.append(img_var)

    std_ = np.sqrt(np.array(var_).mean(axis=0))

    return mean_, std_, min_, max_


mean = torch.tensor([0.32856414, 0.0368233, 0.02034278])
std = torch.tensor([0.23836923, 0.08061118, 0.08756524])


def normalize(image, batch=False):
    """
    Normalize the image in Lab space.
    :param batch:
    :param image:
    :return normalized image:
    """
    transform = transforms.Normalize(mean, std)
    if batch:
        for nn in range(image.shape[0]):
            image[nn, :, :, :] = transform(image[nn, :, :, :])
    else:
        image = transform(image)
    return image


def denormalize(image, CHW=True):
    """
    :param image: {N, H, W, C} {N, C, H, W}
    :param CHW: or HWC?
    :return: denormalized image
    """
    if CHW:
        _std = std.view(3, 1, 1).to(image.device)
        _mean = mean.view(3, 1, 1).to(image.device)
    else:
        _std = std.view(1, 1, 3).to(image.device)
        _mean = mean.view(1, 1, 3).to(image.device)
    return (image * _std) + _mean


if __name__ == '__main__':
    opt = ArgumentParser().parse_args()
    opt.dataroot = "../../CS470_Project/data/safebooru/upper_body_768"
    opt.resize_or_crop = "resize_and_crop"
    opt.loadSize = 512
    opt.fineSize = 512
    opt.isTrain = False
    opt.l_norm = 100.
    opt.l_cent = 50.
    opt.ab_norm = 110.
    opt.sample_Ps = [1, 2, 3, 4, 5, 6, 7, 8, 9, ]
    opt.mask_cent = 0.
    opt.batchSize = 1
    opt.serial_batches = True
    opt.nThreads = 0
    opt.max_dataset_size = 1
    print(get_normalize_value(opt))
