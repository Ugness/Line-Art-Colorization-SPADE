from keras.models import load_model
from keras.engine import Input, Model
import cv2
import numpy as np
from helper import *
import os
import argparse
from tqdm import tqdm

mod = load_model('mod.h5')
# mod.summary()
mod.layers.pop(0)
weights = mod.get_weights()

# mod.summary()
new_input = Input(shape=(None, None, 1))
new_output = mod(new_input)
new_mod = Model(new_input, new_output)
new_mod.summary()
new_mod.layers[-1].set_weights(weights)

def resize_image(size, img):
    pass


def resize_line_mat(size, img):
    pass


def get(path):
    color_path = os.path.join(path, 'color')
    sketch_path = dict()
    sketch_path['pured'] = os.path.join(path, 'pured')
    sketch_path['enhanced'] = os.path.join(path, 'enhanced')
    sketch_path['original'] = os.path.join(path, 'original')
    for opt in sketch_path.keys():
        os.makedirs(sketch_path[opt], exist_ok=True)
    img_list = os.listdir(color_path)
    for img_name in tqdm(img_list):
        from_mat = cv2.imread(os.path.join(color_path, img_name))
        if from_mat is None:
            print(img_name)
            continue
        width = float(from_mat.shape[1])
        height = float(from_mat.shape[0])
        new_width = 0
        new_height = 0
        if width > height:
            from_mat = cv2.resize(from_mat, (768, int(768 / width * height)), interpolation=cv2.INTER_AREA)
            new_width = 768
            new_height = int(768 / width * height)
        else:
            from_mat = cv2.resize(from_mat, (int(768 / height * width), 768), interpolation=cv2.INTER_AREA)
            new_width = int(768 / height * width)
            new_height = 768
        from_mat = from_mat.transpose((2, 0, 1))
        light_map = np.zeros(from_mat.shape, dtype=np.float)
        for channel in range(3):
            light_map[channel] = get_light_map_single(from_mat[channel])
        light_map = normalize_pic(light_map)
        light_map = resize_img_768_3d(light_map)
        line_mat = new_mod.predict(light_map, batch_size=1)
        line_mat = line_mat.transpose((3, 1, 2, 0))[0]
        line_mat = line_mat[0:int(new_height), 0:int(new_width), :]
        line_mat = np.amax(line_mat, 2)
        get_active_img_and_save_denoise_filter2(line_mat,
                                                os.path.join(sketch_path['enhanced'], img_name.replace('png', 'jpg')))
        get_active_img_and_save_denoise_filter(line_mat,
                                               os.path.join(sketch_path['pured'], img_name.replace('png', 'jpg')))
        get_active_img_and_save_denoise(line_mat, os.path.join(sketch_path['original'], img_name.replace('png', 'jpg')))
    return


get(os.path.join('..', '..', 'data', 'safebooru', 'upper_body'))
