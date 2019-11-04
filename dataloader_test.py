from argparse import ArgumentParser

import torchvision.transforms as transforms

import utils.util as util
from dataloader import SafebooruDataLoader

opt = ArgumentParser().parse_args()
opt.dataroot = "../CS470_Project/data/safebooru/upper_body_768"
opt.resize_or_crop = "resize_and_crop"
opt.loadSize = 512
opt.fineSize = 512
opt.isTrain = False
opt.l_norm = 100.
opt.l_cent = 50.
opt.ab_norm = 110.
opt.sample_Ps = [1, 2, 3, 4, 5, 6, 7, 8, 9, ]
opt.mask_cent = 0.
opt.batch_size = 1
opt.serial_batches = True
opt.num_threads = 0
opt.max_dataset_size = 1

loader = SafebooruDataLoader(opt)

def save_image(image, path):
    unloader = transforms.ToPILImage()
    image = util.lab2rgb(image, opt).squeeze(0)
    image = unloader(image)
    image.save(path)

for a, b in enumerate(loader):
    print(b['input'].shape)
    print(b['target'].shape)
    #print((b['input'][:,[1,],:,:] != 0.).sum())
    save_image(b['input'][:, 2:5, :, :], 'Point.jpg')
    save_image(b['target'], 'Output.jpg')
