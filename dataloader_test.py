from argparse import ArgumentParser

import torchvision.transforms as transforms
from PIL import Image

import utils.normalize as normalize
import utils.util as util
from dataloader import SafebooruDataLoader
from dataloader.base_dataset import get_transform

opt = ArgumentParser().parse_args()
opt.dataroot = "../CS470_Project/data/safebooru/upper_body_768"
opt.preprocess_mode = "resize_and_crop"
opt.loadSize = 512
opt.fineSize = 512
opt.isTrain = True
opt.no_flip = False
opt.l_norm = 100.
opt.l_cent = 50.
opt.ab_norm = 110.
opt.sample_Ps = [1, 2, 3, 4, 5, 6, 7, 8, 9, ]
opt.mask_cent = 0.
opt.batchSize = 1
opt.serial_batches = True
opt.nThreads = 0
opt.max_dataset_size = 2

loader = SafebooruDataLoader(opt)

def save_image(image, path):
    unloader = transforms.ToPILImage()
    image = unloader(image)
    image.save(path)

color_path = "../CS470_Project/data/safebooru/upper_body_768/color/1116416.jpg"
color_img = Image.open(color_path).convert('RGB')
color_img = get_transform(opt)(color_img).unsqueeze(0)
img = util.lab2rgb(normalize.denormalize(normalize.normalize(util.rgb2lab(color_img, opt), batch=True)), opt)
img = util.lab2rgb(img, opt).squeeze(0)
save_image(img, "temp.jpg")

#print(color_img.shape)

for a, b in enumerate(loader):
    #print(b['label'].shape)
    #print(b['target'].shape)

    #save_image(b['input'][:, 2:5, :, :], 'Point.jpg')

    image = normalize.denormalize(b['label'])
    image = util.lab2rgb(image, opt).squeeze(0)
    save_image(image, 'label.jpg')
    save_image(b['image'].squeeze(0), 'image.jpg')

