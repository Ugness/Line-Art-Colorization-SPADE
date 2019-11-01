import torch
from torchvision import transforms
from torchvision.utils import save_image
from torch import nn

from PIL import Image
import argparse

parser = argparse.ArgumentParser(description='Sketch simplification demo.')
parser.add_argument('--model', type=str, default='model.pth', help='Model to use.')
parser.add_argument('--img',   type=str, default='test.png',     help='Input image file.')
parser.add_argument('--out',   type=str, default='out.png',      help='File to output.')
opt = parser.parse_args()

use_cuda = torch.cuda.device_count() > 0

model = nn.Sequential( # Sequential,
	nn.Conv2d(1,48,(5, 5),(2, 2),(2, 2)),
	nn.ReLU(),
	nn.Conv2d(48,128,(3, 3),(1, 1),(1, 1)),
	nn.ReLU(),
	nn.Conv2d(128,128,(3, 3),(1, 1),(1, 1)),
	nn.ReLU(),
	nn.Conv2d(128,128,(3, 3),(2, 2),(1, 1)),
	nn.ReLU(),
	nn.Conv2d(128,256,(3, 3),(1, 1),(1, 1)),
	nn.ReLU(),
	nn.Conv2d(256,256,(3, 3),(1, 1),(1, 1)),
	nn.ReLU(),
	nn.Conv2d(256,256,(3, 3),(2, 2),(1, 1)),
	nn.ReLU(),
	nn.Conv2d(256,512,(3, 3),(1, 1),(1, 1)),
	nn.ReLU(),
	nn.Conv2d(512,1024,(3, 3),(1, 1),(1, 1)),
	nn.ReLU(),
	nn.Conv2d(1024,1024,(3, 3),(1, 1),(1, 1)),
	nn.ReLU(),
	nn.Conv2d(1024,1024,(3, 3),(1, 1),(1, 1)),
	nn.ReLU(),
	nn.Conv2d(1024,1024,(3, 3),(1, 1),(1, 1)),
	nn.ReLU(),
	nn.Conv2d(1024,512,(3, 3),(1, 1),(1, 1)),
	nn.ReLU(),
	nn.Conv2d(512,256,(3, 3),(1, 1),(1, 1)),
	nn.ReLU(),
	nn.ConvTranspose2d(256,256,(4, 4),(2, 2),(1, 1),(0, 0)),
	nn.ReLU(),
	nn.Conv2d(256,256,(3, 3),(1, 1),(1, 1)),
	nn.ReLU(),
	nn.Conv2d(256,128,(3, 3),(1, 1),(1, 1)),
	nn.ReLU(),
	nn.ConvTranspose2d(128,128,(4, 4),(2, 2),(1, 1),(0, 0)),
	nn.ReLU(),
	nn.Conv2d(128,128,(3, 3),(1, 1),(1, 1)),
	nn.ReLU(),
	nn.Conv2d(128,48,(3, 3),(1, 1),(1, 1)),
	nn.ReLU(),
	nn.ConvTranspose2d(48,48,(4, 4),(2, 2),(1, 1),(0, 0)),
	nn.ReLU(),
	nn.Conv2d(48,24,(3, 3),(1, 1),(1, 1)),
	nn.ReLU(),
	nn.Conv2d(24,1,(3, 3),(1, 1),(1, 1)),
	nn.Sigmoid(),
)
model.load_state_dict(torch.load('model.pth'))
immean = 0.9664114577640158
imstd = 0.0858381272736797

data  = Image.open( opt.img ).convert('L')
w, h  = data.size[0], data.size[1]
pw    = 8-(w%8) if w%8!=0 else 0
ph    = 8-(h%8) if h%8!=0 else 0
data  = ((transforms.ToTensor()(data)-immean)/imstd).unsqueeze(0)
if pw!=0 or ph!=0:
   data = torch.nn.ReplicationPad2d( (0,pw,0,ph) )( data ).data

if use_cuda:
   pred = model.cuda().forward( data.cuda() ).float()
else:
   pred = model.forward( data )
save_image( pred[0], opt.out )


