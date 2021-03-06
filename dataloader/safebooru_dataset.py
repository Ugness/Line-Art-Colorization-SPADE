import os.path

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import transforms
import torchvision.transforms.functional as F
from utils.normalize import normalize

import utils.util as util
from dataloader.base_dataset import BaseDataset, get_transform
from dataloader.image_folder import make_dataset
import random

class SafebooruDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.set_defaults(fineSize=256)
        parser.set_defaults(loadSize=256)
        parser.set_defaults(crop_size=256)
        parser.set_defaults(preprocess_mode='scale_width')
        parser.set_defaults(label_nc=5)

        parser.add_argument('--l_norm', type=float, default=100.)
        parser.add_argument('--l_cent', type=float, default=50.)
        parser.add_argument('--ab_norm', type=float, default=110.)
        parser.add_argument('--sample_Ps', type=int, nargs=3, default=[1, 9, 1])
        parser.add_argument('--mask_cent', type=float, default=0.)
        parser.add_argument('--hsv_aug', type=float, default=0, help='Use HSV Augmentation')

        return parser

    def initialize(self, opt):
        self.opt = opt
        self.opt.sample_Ps = range(*opt.sample_Ps)
        self.root = opt.dataroot

        self.color_dir = os.path.join(opt.dataroot, 'color')
        self.color_paths = make_dataset(self.color_dir)
        self.color_paths = sorted(self.color_paths)

        folder = ['line']
        self.category = os.listdir(os.path.join(opt.dataroot, 'line'))
        idx1 = 0
        self.line_dir = dict()
        self.line_paths = dict()
        for key in self.category:
            dir = os.path.join(folder[idx1], key)
            self.line_dir[key] = os.path.join(opt.dataroot, dir)
            self.line_paths[key] = make_dataset(self.line_dir[key])
            self.line_paths[key] = sorted(self.line_paths[key])

        self.transform = get_transform(opt)
        if self.opt.hsv_aug > 0:
            self.hsv_augmenter = transforms.ColorJitter(hue=(-self.opt.hsv_aug, self.opt.hsv_aug))

    @staticmethod
    def image_name(path):
        return os.path.splitext(os.path.basename(path))[0]

    def paths_match(self, path1, path2):
        filename1_without_ext = self.image_name(path1)
        filename2_without_ext = self.image_name(path2)
        return filename1_without_ext == filename2_without_ext

    def sync_transform(self, color_img, line_img):
        color_w, color_h = color_img.size
        line_w, line_h = line_img.size

        n = min(color_w, color_h, line_w, line_h)

        color_img = F.resize(color_img, (n, n))  # Size of n*n*3
        if self.opt.hsv_aug:
            color_img = self.hsv_augmenter(color_img)
        line_img = F.resize(line_img, (n, n), Image.NEAREST)
        line_img = np.expand_dims(np.array(line_img), axis=2)  # Size of n*n*1
        transformed_img = self.transform(Image.fromarray(np.concatenate([color_img, line_img], axis=2)))

        color_img = transformed_img[0:3, :, :].unsqueeze(0)  # Size of 1*3*m*m
        line_img = transformed_img[[3, ], :, :].unsqueeze(0)  # Size of 1*1*m*m

        return color_img, line_img

    def __getitem__(self, index):
        # index = index % 100
        color_path = self.color_paths[index]
        color_img = Image.open(color_path).convert('RGB')
        key = random.choice(self.category)
        line_path = self.line_paths[key][index]
        line_img = Image.open(line_path).convert('L')

        assert self.paths_match(color_path, line_path), \
            "The label_path %s and image_path %s don't match." % \
            (color_path, line_path)


        color_img, line_img = self.sync_transform(color_img, line_img)

        color_img = normalize(color_img)
        line_img = normalize(line_img)

        # Target tensor: normalized Lab color image
        target_tensor = color_img.squeeze(0)

        colorization_data = util.get_colorization_data(color_img, self.opt)

        # Fit to SPADE
        real_image = target_tensor
        hint_tensor = torch.cat((colorization_data['mask_B'],
                                     colorization_data['hint_B']), dim=1).squeeze(0)
        sketch_tensor = line_img.squeeze(0)
        image_path = line_path

        return {'label': real_image,
                'instance': hint_tensor,
                'image': sketch_tensor,
                'path': image_path,
                }

    def __len__(self):
        return len(self.color_paths)

    def name(self):
        return 'SafebooruDataset'
