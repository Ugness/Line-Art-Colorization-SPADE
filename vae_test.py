"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from torchvision.utils import save_image

import dataloader
from options.demo_options import DemoOptions
from models.pix2pix_model import Pix2PixModel
from demo import process

opt = DemoOptions().parse()

dataloader = dataloader.SafebooruDataLoader(opt)

device = 'cpu' if opt.gpu_ids == -1 else 'cuda'
model = Pix2PixModel(opt)
model.eval()
np_shifts = np.arange(-100, 100, 10)
single_data = dataloader.dataset[0]
data = dict()
for key, value in single_data.items():
    if isinstance(value, torch.Tensor):
        data[key] = torch.stack([value for _ in np_shifts], dim=0)
shifts_1dim = torch.tensor(np_shifts).to(device)
# test
fakes = []
for i in tqdm(range(opt.z_dim//10)):
    shifts = torch.zeros([len(np_shifts), opt.z_dim]).to(device)
    shifts[:, i] = shifts_1dim
    data['shift'] = shifts
    with torch.no_grad():
        generated = model(data, mode='demo')
        F.interpolate(generated, size=32)
        fakes.append(process.denormalize(generated).detach().cpu().data)
fakes = torch.cat(fakes, dim=0)
save_image(fakes, './test.png', nrow=len(np_shifts))
