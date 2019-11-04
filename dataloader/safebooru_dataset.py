import os.path

import torch
from PIL import Image

import utils.normalize as normalize
import utils.util as util
from dataloader.base_dataset import BaseDataset, get_transform
from dataloader.image_folder import make_dataset


class SafebooruDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot

        self.color_dir = os.path.join(opt.dataroot, 'color/')
        self.color_paths = make_dataset(self.color_dir)
        self.color_paths = sorted(self.color_paths)

        folder = ['line/', 'sketch/']
        category = ['enhanced/', 'original/', 'pured/']
        idx1 = (torch.randint(0, 2, (1, ))).item()
        idx2 = (torch.randint(0, 3, (1, ))).item()
        dir = folder[idx1] + category[idx2]

        self.line_dir = os.path.join(opt.dataroot, dir)
        self.line_paths = make_dataset(self.line_dir)
        self.line_paths = sorted(self.line_paths)

        self.transform = get_transform(opt)

    @staticmethod
    def image_name(path):
        return os.path.splitext(os.path.basename(path))[0]

    def paths_match(self, path1, path2):
        filename1_without_ext = self.image_name(path1)
        filename2_without_ext = self.image_name(path2)
        return filename1_without_ext == filename2_without_ext

    def __getitem__(self, index):
        color_path = self.color_paths[index]
        color_img = Image.open(color_path).convert('RGB')
        color_img = self.transform(color_img).unsqueeze(0)

        line_path = self.line_paths[index]
        line_img = Image.open(line_path).convert('L')
        line_img = self.transform(line_img).unsqueeze(0)

        assert self.paths_match(color_path, line_path), \
            "The label_path %s and image_path %s don't match." % \
            (color_path, line_path)

        # Target tensor: Lab color image
        target_tensor = normalize.normalize(util.rgb2lab(color_img, self.opt).squeeze(0))

        colorization_data = util.get_colorization_data(color_img, self.opt)

        # Input tensor: Line image + mask (1 channel) + hint (Lab from Lab)
        input_tensor = torch.cat((line_img,
                                  colorization_data['mask_B'],
                                  colorization_data['hint_B']), dim=1).squeeze(0)

        return {'input': input_tensor,
                'target': target_tensor,
                'color_path': color_path,
                'line_path': line_path,
                }

    def __len__(self):
        return len(self.color_paths)

    def name(self):
        return 'SafebooruDataset'
