import os
import io
from time import gmtime, strftime
import base64
import argparse
import random

import numpy as np
import torch
import torchvision.transforms.functional as F
from PIL import Image
from flask import Flask, flash, redirect, render_template, request, session, abort, jsonify

import process

import sys

sys.path.append('..')

from options.demo_options import DemoOptions
from models.pix2pix_model import Pix2PixModel
from utils.preprocessing.sketch_simplification.simplify import get_model
from util import util

torch.backends.cudnn.deterministic = True

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
    hint_name = './data/hint/hint_' + timestamp + '.png'
    with open(hint_name, 'wb') as f:
        f.write(hintdata)
        f.close()

    line = request.form.get("line")
    linedata = base64.b64decode(line.split(',')[1])
    line_name = './data/line/line_' + timestamp + '.png'
    with open(line_name, 'wb') as f:
        f.write(linedata)
        f.close()

    output_name = './data/output/output_' + timestamp + '.png'

    # line = Image.open('../../CS470_Project/data/safebooru/test_upper_body_768/line/original/2329616.jpg').convert('L')
    hint = Image.open(io.BytesIO(hintdata)).convert("RGBA")
    line = Image.open(io.BytesIO(linedata)).convert("L")
    line.save(line_name)
    hint.save(hint_name)

    hint = np.array(hint)  # RGBA

    line = np.array(line)  # RGBA -> L
    line = np.expand_dims(line, axis=2)

    # RGBA to mask + hint
    mask = hint[:, :, [3, ]]
    hint = hint[:, :, [0, 1, 2]]

    # Resize
    line = process.resize_img(line, size=256, pad_value=255, mode='nearest')
    hint = process.resize_img(hint, size=256, pad_value=0, mode='bilinear')
    mask = process.resize_img(mask, size=256, pad_value=0, mode='bilinear')

    hint = np.concatenate([hint, mask], axis=2)

    data = process.getitem(line, hint)
    data['shift'] = z
    with torch.no_grad():
        color_img = model(data, 'demo')
    color_img = util.tensor2im(color_img)
    util.save_image(color_img[0], output_name, create_dir=True)

    # for test
    f_name = output_name
    with open(f_name, 'rb') as f:
        output = base64.b64encode(f.read()).decode("utf-8")
        f.close()

    data = {'output': prefix + "," + output}
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

    f_name = './data/line/line_' + timestamp + '.png'
    with open(f_name, 'wb') as f:
        f.write(imgdata)
        f.close()

    immean = 0.9664114577640158
    imstd = 0.0858381272736797
    img = Image.open(io.BytesIO(imgdata)).convert("L")
    img = np.expand_dims(np.array(img), axis=2)
    data = process.resize_img(img, size=768, mode='nearest')
    print(data.shape)
    data = ((F.to_tensor(data) - immean) / imstd).unsqueeze(0).to(device)
    pred = sketch_model.forward(data)
    f_name = './data/simplified/simplified_' + timestamp + '.png'
    F.to_pil_image(pred[0].cpu()).save(f_name)
    with open(f_name, 'rb') as f:
        output = base64.b64encode(f.read()).decode("utf-8")
        f.close()

    data = {'output': prefix + "," + output}
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

    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed_all(opt.seed)
    np.random.seed(opt.seed)
    random.seed(opt.seed)

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
