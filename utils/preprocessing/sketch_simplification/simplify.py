import torch
from torchvision import transforms
from torchvision.utils import save_image
from torch import nn
from PIL import Image
import argparse
import os
import sys
import datetime
from tqdm import tqdm

immean = 0.9664114577640158
imstd = 0.0858381272736797


def get_model():
    return nn.Sequential(  # Sequential,
        nn.Conv2d(1, 48, (5, 5), (2, 2), (2, 2)),
        nn.ReLU(),
        nn.Conv2d(48, 128, (3, 3), (1, 1), (1, 1)),
        nn.ReLU(),
        nn.Conv2d(128, 128, (3, 3), (1, 1), (1, 1)),
        nn.ReLU(),
        nn.Conv2d(128, 128, (3, 3), (2, 2), (1, 1)),
        nn.ReLU(),
        nn.Conv2d(128, 256, (3, 3), (1, 1), (1, 1)),
        nn.ReLU(),
        nn.Conv2d(256, 256, (3, 3), (1, 1), (1, 1)),
        nn.ReLU(),
        nn.Conv2d(256, 256, (3, 3), (2, 2), (1, 1)),
        nn.ReLU(),
        nn.Conv2d(256, 512, (3, 3), (1, 1), (1, 1)),
        nn.ReLU(),
        nn.Conv2d(512, 1024, (3, 3), (1, 1), (1, 1)),
        nn.ReLU(),
        nn.Conv2d(1024, 1024, (3, 3), (1, 1), (1, 1)),
        nn.ReLU(),
        nn.Conv2d(1024, 1024, (3, 3), (1, 1), (1, 1)),
        nn.ReLU(),
        nn.Conv2d(1024, 1024, (3, 3), (1, 1), (1, 1)),
        nn.ReLU(),
        nn.Conv2d(1024, 512, (3, 3), (1, 1), (1, 1)),
        nn.ReLU(),
        nn.Conv2d(512, 256, (3, 3), (1, 1), (1, 1)),
        nn.ReLU(),
        nn.ConvTranspose2d(256, 256, (4, 4), (2, 2), (1, 1), (0, 0)),
        nn.ReLU(),
        nn.Conv2d(256, 256, (3, 3), (1, 1), (1, 1)),
        nn.ReLU(),
        nn.Conv2d(256, 128, (3, 3), (1, 1), (1, 1)),
        nn.ReLU(),
        nn.ConvTranspose2d(128, 128, (4, 4), (2, 2), (1, 1), (0, 0)),
        nn.ReLU(),
        nn.Conv2d(128, 128, (3, 3), (1, 1), (1, 1)),
        nn.ReLU(),
        nn.Conv2d(128, 48, (3, 3), (1, 1), (1, 1)),
        nn.ReLU(),
        nn.ConvTranspose2d(48, 48, (4, 4), (2, 2), (1, 1), (0, 0)),
        nn.ReLU(),
        nn.Conv2d(48, 24, (3, 3), (1, 1), (1, 1)),
        nn.ReLU(),
        nn.Conv2d(24, 1, (3, 3), (1, 1), (1, 1)),
        nn.Sigmoid(),
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Sketch simplification demo.')
    parser.add_argument('--img', type=str, default='input', help='Directory containing input image files.')
    parser.add_argument('--out', type=str, default='output', help='Directory to save output image files.')
    opt = parser.parse_args()

    use_cuda = torch.cuda.device_count() > 0
    if use_cuda:
        device = "cuda:0"
    else:
        device = "cpu"

    model = get_model()

    model.load_state_dict(torch.load('model.pth'))
    if use_cuda:
        model = model.cuda()
    else:
        model = model.cpu()

    img_dir = opt.img
    out_dir = opt.out
    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    it = 0
    for img_name in tqdm(os.listdir(img_dir)):
        it += 1
        # if it % 100 == 0:
        #     print(f"[{it:04d}]{img_name} {datetime.datetime.now()}")

        try:
            img_path = os.path.join(img_dir, img_name)
            data = Image.open(img_path).convert('L')
        except:
            print(f"error loading {img_name}")
            continue
        w, h = data.size[0], data.size[1]
        pw = 8 - (w % 8) if w % 8 != 0 else 0
        ph = 8 - (h % 8) if h % 8 != 0 else 0
        data = ((transforms.ToTensor()(data) - immean) / imstd).unsqueeze(0).to(device)
        with torch.no_grad():
            if pw != 0 or ph != 0:
                data = torch.nn.ReplicationPad2d((0, pw, 0, ph))(data).data
            pred = model.forward(data)

            convert2img = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(512)
            ])
            out_path = os.path.join(out_dir, img_name)
            convert2img(pred[0].cpu()).save(out_path)
