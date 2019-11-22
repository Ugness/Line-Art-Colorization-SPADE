from flask import Flask, flash, redirect, render_template, request, session, abort, jsonify
import base64
import numpy as np
import argparse
from time import gmtime, strftime
import torch
import os
import process
import torchvision.transforms.functional as F
import cv2
from PIL import Image
import io

import sys
sys.path.append('..')

from options.demo_options import DemoOptions
from models.pix2pix_model import Pix2PixModel
from utils.preprocessing.sketch_simplification.simplify import get_model
from util import util

app = Flask(__name__)


@app.route("/")
def index():
    return render_template('test.html')

  
@app.route("/colorization/", methods=['POST'])
def sum():
    # Preparing Inputs
    rgba = request.form.get("rgba")
    width = int(float(request.form.get("width")))
    height = int(float(request.form.get("height")))
    z = float(request.form.get("z"))

    hintdata = base64.b64decode(rgba.split(',')[1])
    prefix = rgba.split(',')[0]
    timestamp = strftime("%Y%m%d%H%M%S", gmtime())
    hint_name = './data/hint/hint_'+ timestamp +'.png'
    with open(hint_name, 'wb') as f:
        f.write(hintdata)
        f.close()

    line = request.form.get("line")
    linedata = base64.b64decode(line.split(',')[1])
    line_name = './data/line/line_' + timestamp + '.png'
    with open(line_name, 'wb') as f:
        f.write(linedata)
        f.close()

    line = Image.open('../../CS470_Project/data/safebooru/test_upper_body_768/line/original/2329616.jpg').convert('L')
    hint = Image.open(io.BytesIO(hintdata)).convert("RGBA")
    # line = Image.open(io.BytesIO(linedata)).convert("L")
    hint.save(hint_name)
    line.save(line_name)

    hint = np.array(hint)   # RGBA

    line = np.array(line)   # RGBA -> L
    line = np.expand_dims(line, axis=2)

    # RGBA to mask + hint
    mask = hint[:, :, [3, ]]
    hint = hint[:, :, [0, 1, 2]]

    # Resize
    line = process.resize_img(line, size=256, pad_value=255, mod='nearest')
    hint = process.resize_img(hint, size=256, pad_value=0, mod='bilinear')
    mask = process.resize_img(mask, size=256, pad_value=0, mod='bilinear')

    # ndarray to torch.Tensor
    line = F.to_tensor(line).float()
    hint = F.to_tensor(hint).float()
    mask = F.to_tensor(mask).float()

    # normalize
    line = process.normalize(line)
    hint = process.normalize(hint)

    # concat Mask and Hint
    # print(hint.max(), mask.max(), line.max())
    # print(hint.min(), mask.min(), line.min())
    mask_hint = torch.cat([mask, hint], dim=0).unsqueeze(0).to(device)
    line = line.unsqueeze(0).to(device)

    F.to_pil_image(process.denormalize(hint.cpu())).save(hint_name)
    F.to_pil_image(mask).save('./data/mask.png')
    F.to_pil_image(process.denormalize(line[0].cpu())).save(line_name)

    data = {'label': None, 'instance': mask_hint, 'image': line}
    with torch.no_grad():
        output = model(data, 'inference')
    print(output.max(), output.min())
    output = util.tensor2im(output)
    util.save_image(output[0], './data/output/output_test.png', create_dir=True)
    print(output.shape)
    # output = F.to_pil_image(output)
    # output.save('./data/output/output_test.png')

    #for test
    f_name = './data/output/output_test.png'
    with open(f_name, 'rb') as f:
        output = base64.b64encode(f.read()).decode("utf-8")
        f.close()

    data = {'output': prefix+","+output}
    data = jsonify(data)
    return data

@app.route("/simplification/", methods=['POST'])
def simplification():
    line = request.form.get("line")
    width = int(float(request.form.get("width")))
    height = int(float(request.form.get("height")))

    imgdata = base64.b64decode(line.split(',')[1])
    prefix = line.split(',')[0]
    timestamp = strftime("%Y%m%d%H%M%S", gmtime())

    f_name = './data/line/line_'+ timestamp +'.png'
    with open(f_name, 'wb') as f:
        f.write(imgdata)
        f.close()

    np_img = np.frombuffer(imgdata)
    print(np_img.shape)
    f_name = './data/simplified/simplified_'+ timestamp +'.png'
    with open(f_name, 'rb') as f:
        output = base64.b64encode(f.read()).decode("utf-8")
        f.close()

    data = {'output': prefix+","+output}
    data = jsonify(data)
    return data


if __name__ == "__main__":
    opt = DemoOptions().parse()
    if opt.name is None:
        print("Experience Name required for loading model")
        exit()
    if opt.gpu_ids != -1:
        device = 'cuda'
    else:
        device = 'cpu'
    print("Loading Models ....")
    model = Pix2PixModel(opt)
    model.eval()
    sketch_model = get_model().to(device)
    sketch_dict = torch.load(opt.simplification, map_location=device)
    sketch_model.load_state_dict(sketch_dict)
    print("Loading Models Done")

    os.makedirs('./data/hint', exist_ok=True)
    os.makedirs('./data/output', exist_ok=True)
    os.makedirs('./data/simplified', exist_ok=True)
    os.makedirs('./data/line', exist_ok=True)

    app.run(host='0.0.0.0', port=opt.port, debug=True)
