"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import os
import sys

from collections import OrderedDict
import torch

import dataloader
from options.test_options import TestOptions
from models.pix2pix_model import Pix2PixModel
from util.visualizer import Visualizer
from util import html

# print(sys.argv)
# print(len(sys.argv))

sys.argv = "compare_test.py --checkpoints_dir ../CS470_Project/checkpoints --name Lnet_hsv_aug_0.1 --dataset_mode safebooru --dataroot ../CS470_Project/data/safebooru/test_upper_body_768 --batchSize 8 --gpu_ids 0 --use_vae --netG spadeladder".split(
    " ")
opt = TestOptions().parse()
sys.argv = "compare_test.py --checkpoints_dir ../CS470_Project/checkpoints --name no_Lnet --dataset_mode safebooru --dataroot ../CS470_Project/data/safebooru/test_upper_body_768 --batchSize 8 --gpu_ids 0 --use_vae --netG spade".split(" ")
opt2 = TestOptions().parse()
opt.sample_Ps = [2, 22, 2]
dataloader = dataloader.SafebooruDataLoader(opt)

model = Pix2PixModel(opt)
model_noLnet = Pix2PixModel(opt2)
model.eval()

visualizer = Visualizer(opt)
vis2 = Visualizer(opt2)

# create a webpage that summarizes the all results
web_dir = os.path.join(opt.results_dir, opt.name,
                       '%s_%s' % (opt.phase, opt.which_epoch))
webpage = html.HTML(web_dir,
                    'Experiment = %s, Phase = %s, Epoch = %s' %
                    (opt.name, opt.phase, opt.which_epoch))

web_dir2 = os.path.join(opt2.results_dir, opt2.name,
                        '%s_%s' % (opt2.phase, opt2.which_epoch))
webpage2 = html.HTML(web_dir2,
                     'Experiment = %s, Phase = %s, Epoch = %s' %
                     (opt2.name, opt2.phase, opt2.which_epoch))

# test
with torch.no_grad():
    for i, data_i in enumerate(dataloader):
        if i * opt.batchSize >= opt.how_many:
            break

        generated = model(data_i, mode='inference')

        img_path = data_i['path']
        hint = data_i['instance'][:, [1, 2, 3, ], :, :]
        mask = (data_i['instance'][:, [0, ], :, :] > 0).float()
        color = data_i['label']
        hint_map = mask * hint + color * 0.3 * (1 - mask)
        for b in range(generated.shape[0]):
            print('process image... %s' % img_path[b])
            visuals = OrderedDict([('input_label', data_i['label'][b]),
                                   ('synthesized_image', generated[b]),
                                   ('hint', hint_map[b])])
            visualizer.save_images(webpage, visuals, img_path[b:b + 1])

        generated = model_noLnet(data_i, mode='inference')

        img_path = data_i['path']
        for b in range(generated.shape[0]):
            print('process image... %s' % img_path[b])
            visuals = OrderedDict([('input_label', data_i['label'][b]),
                                   ('synthesized_image', generated[b]),
                                   ('hint', hint_map[b])])
            vis2.save_images(webpage2, visuals, img_path[b:b + 1])

webpage.save()
webpage2.save()
