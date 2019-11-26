import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
import io
from time import gmtime, strftime
import base64
import random

# modules for sketchKeras
import tensorflow as tf
from keras.models import load_model
from keras.engine import Input, Model

from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.1
set_session(tf.Session(config=config))

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
from utils.sketchKeras.helper import *
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
    isDeter = float(request.form.get("isDeter")) > 0.5
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
    if not isDeter:
        data['shift'] = 'random'
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

@app.route("/default/", methods=['POST'])
def default():
    f_name = './data/default/default_yoomi.png'
    with open(f_name, 'rb') as f:
        output = base64.b64encode(f.read()).decode("utf-8")
        f.close()

    data = {'output':"data:image/png;base64," + output}
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
    img = np.array(Image.open(io.BytesIO(imgdata)).convert("RGB"))
    # img = np.expand_dims(img, axis=2)
    img = process.resize_img(img, size=768, mode='bilinear')
    from_mat = img
    H, W, C = img.shape
    from_mat = from_mat.transpose((2, 0, 1))  # HWC -> CHW
    light_map = np.zeros(from_mat.shape, dtype=np.float)
    for channel in range(3):
        light_map[channel] = get_light_map_single(from_mat[channel])
    light_map = normalize_pic(light_map)
    light_map = resize_img_768_3d(light_map)
    with graph.as_default():
        line_mat = new_mod.predict(light_map, batch_size=1)
    line_mat = line_mat.transpose((3, 1, 2, 0))[0]
    line_mat = line_mat[0:H, 0:W, :]
    line_mat = np.amax(line_mat, 2)
    sketch = get_enhanced_img(line_mat)
    print(sketch.shape)

    print(F.to_tensor(sketch).max())
    with torch.no_grad():
        sketch = ((F.to_tensor(sketch) - immean) / imstd).unsqueeze(0).to(device)
        pred = simp_model.forward(sketch)
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
    simp_model = get_model().to(device)
    simp_state_dict = torch.load(opt.simplification, map_location=device)
    simp_model.load_state_dict(simp_state_dict)

    mod = load_model(opt.sketch)
    mod.layers.pop(0)
    weights = mod.get_weights()

    new_input = Input(shape=(None, None, 1))
    new_output = mod(new_input)
    new_mod = Model(new_input, new_output)
    new_mod.layers[-1].set_weights(weights)
    graph = tf.get_default_graph()

    print("Loading Models Done")
    os.makedirs('./data/hint', exist_ok=True)
    os.makedirs('./data/output', exist_ok=True)
    os.makedirs('./data/simplified', exist_ok=True)
    os.makedirs('./data/line', exist_ok=True)

    app.run(host='0.0.0.0', port=opt.port, debug=True)
