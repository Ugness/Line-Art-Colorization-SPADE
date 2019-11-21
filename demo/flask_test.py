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

import sys
sys.path.append('..')

from options.demo_options import DemoOptions
from models.pix2pix_model import Pix2PixModel
from utils.preprocessing.sketch_simplification.simplify import get_model

app = Flask(__name__)


@app.route("/")
def index():
    return render_template('test.html')

  
@app.route("/colorization/", methods=['POST'])
def sum():
    rgba = request.form.get("rgba")
    width = int(float(request.form.get("width")))
    height = int(float(request.form.get("height")))
    z = float(request.form.get("z"))

    hintdata = base64.b64decode(rgba.split(',')[1])
    prefix = rgba.split(',')[0]
    timestamp = strftime("%Y%m%d%H%M%S", gmtime())
    f_name = './data/hint/hint_'+ timestamp +'.png'
    with open(f_name, 'wb') as f:
        f.write(hintdata)
        f.close()

    # line = request.form.get("line")
    # linedata = base64.b64decode(line.split(',')[1])
    # line = np.frombuffer(linedata, dtype=np.uint8)
    line_path = '../../CS470_Project/data/safebooru/solo/line/original'
    line_img = os.listdir(line_path)
    line = cv2.imread(os.path.join(line_path, line_img[0]))
    hint = cv2.imread(f_name, flags=cv2.IMREAD_UNCHANGED)
    # print(line.shape, hint.shape)
    print(hint.shape)
    line = np.zeros([256, 256, 3])

    line = process.resize_img(line, size=256, pad_value=255)
    hint = process.resize_img(hint, size=256, pad_value=0)
    line = np.sum(line, axis=2, keepdims=True)

    line = F.to_tensor(line).unsqueeze(0).to(device).float()
    line = process.normalize(line)
    hint = F.to_tensor(hint).unsqueeze(0).to(device).float()
    hint[:, [0, 1, 2], :, :] = process.normalize(hint[:, [0, 1, 2], :, :])
    input = torch.cat([line, hint], dim=1)[:, [0, 4, 1, 2, 3], :, :]  # sketch, Mask, RGB
    with torch.no_grad():
        output = process.denormalize(model.inference(input, z))
    output = output.squeeze(0).cpu().numpy()
    output = np.transpose(output, axes=[1, 2, 0])
    print(output.shape)
    print(output.min(), output.max(), output.mean())
    cv2.imwrite('./data/output/output_test.png', output)

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
    if opt.cuda:
        device = 'cuda'
    else:
        device = 'cpu'
    print("Loading Models ....")
    model = Pix2PixModel(opt).to(device)
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
