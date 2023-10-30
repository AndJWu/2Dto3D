import copy
import os

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf

from parameters import (BATCH_SIZE, BIN, MAX_JIT, NORM_H, NORM_W,
                        OVERLAP, VEHICLES, image_dir, label_dir)


def compute_anchors(angle):
    anchors = []

    spaceangle = 2*np.pi / BIN
    l_index = int(angle/spaceangle)
    r_index = l_index + 1

    if (angle - l_index*spaceangle) < spaceangle/2 * (1 + OVERLAP/2):
        anchors.append([l_index, angle - l_index*spaceangle])

    if (r_index*spaceangle - angle) < spaceangle/2 * (1+OVERLAP/2):
        anchors.append([r_index % BIN, angle - r_index*spaceangle])

    return anchors


def parse_annotation(label_dir, image_dir):
    all_objs = []
    dims_avg = {key: np.array([0, 0, 0]) for key in VEHICLES}
    dims_cnt = {key: 0 for key in VEHICLES}

    for label_file in os.listdir(label_dir):
        image_file = label_file.replace('txt', 'png')

        for line in open(label_dir + label_file).readlines():
            line = line.strip().split(' ')
            truncated = np.abs(float(line[1]))
            occluded = np.abs(float(line[2]))

            if line[0] in VEHICLES and truncated < 0.1 and occluded < 0.1:
                new_alpha = float(line[3]) + np.pi/2.
                if new_alpha < 0:
                    new_alpha = new_alpha + 2.*np.pi
                new_alpha = new_alpha - int(new_alpha/(2.*np.pi))*(2.*np.pi)

                obj = {'name': line[0],
                       'image': image_file,
                       'xmin': int(float(line[4])),
                       'ymin': int(float(line[5])),
                       'xmax': int(float(line[6])),
                       'ymax': int(float(line[7])),
                       'dims': np.array([float(number) for number in line[8:11]]),
                       'new_alpha': new_alpha
                       }

                dims_avg[obj['name']] = dims_cnt[obj['name']] * \
                    dims_avg[obj['name']] + obj['dims']
                dims_cnt[obj['name']] += 1
                dims_avg[obj['name']] /= dims_cnt[obj['name']]

                all_objs.append(obj)

    return all_objs, dims_avg


all_objs, dims_avg = parse_annotation(label_dir, image_dir)

for obj in all_objs:
    # Fix dimensions
    obj['dims'] = obj['dims'] - dims_avg[obj['name']]

    # Fix orientation and confidence for no flip
    orientation = np.zeros((BIN, 2))
    confidence = np.zeros(BIN)

    anchors = compute_anchors(obj['new_alpha'])

    for anchor in anchors:
        orientation[anchor[0]] = np.array(
            [np.cos(anchor[1]), np.sin(anchor[1])])
        confidence[anchor[0]] = 1

    confidence = confidence / np.sum(confidence)

    obj['orient'] = orientation
    obj['conf'] = confidence


def prepare_input_and_output(train_inst):
    # Prepare image patch
    xmin = int(train_inst['xmin'])
    ymin = int(train_inst['ymin'])
    xmax = int(train_inst['xmax'])
    ymax = int(train_inst['ymax'])

    img = cv2.imread(image_dir + train_inst['image'])
    img = copy.deepcopy(img[ymin:ymax+1, xmin:xmax+1]).astype(np.float32)

    # re-color the image
    img += np.random.randint(-2, 3, img.shape).astype('float32')
    t = [np.random.uniform(), np.random.uniform(), np.random.uniform()]
    t = np.array(t)

    img = img * (1 + t)
    img = img / (255 * 2)

    # Add rotation (rotate between -20 and 20 degrees)
    angle = np.random.uniform(-20, 20)
    M = cv2.getRotationMatrix2D((img.shape[1]/2, img.shape[0]/2), angle, 1)
    # Check if image dimensions are valid
    if img.shape[0] <= 0 or img.shape[1] <= 0:
        print(f"Invalid image dimensions: {img.shape}, skipping this image.")
        # Skip this image, or handle this error in a way suitable for your case
        return None, None, None, None
    img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))

    # Add scaling
    scale_factor = np.random.uniform(0.8, 1.2)
    img = cv2.resize(img, None, fx=scale_factor, fy=scale_factor)

    # Update position and dimensions according to scaling
    train_inst['xmin'] *= scale_factor
    train_inst['ymin'] *= scale_factor
    train_inst['xmax'] *= scale_factor
    train_inst['ymax'] *= scale_factor
    train_inst['dims'] *= scale_factor

    # Compute center and relative position to the center
    center = [img.shape[1]/2, img.shape[0]/2]
    pos = [(train_inst['xmax']+train_inst['xmin'])/2, (train_inst['ymax']+train_inst['ymin'])/2]
    rel_pos = [pos[0]-center[0], pos[1]-center[1]]

    # Compute new position after rotation
    new_rel_pos = [rel_pos[0]*np.cos(np.deg2rad(angle)) - rel_pos[1]*np.sin(np.deg2rad(angle)),
                   rel_pos[0]*np.sin(np.deg2rad(angle)) + rel_pos[1]*np.cos(np.deg2rad(angle))]
    new_pos = [new_rel_pos[0]+center[0], new_rel_pos[1]+center[1]]

    # Update position
    train_inst['xmin'] = new_pos[0] - train_inst['dims'][0] / 2
    train_inst['xmax'] = new_pos[0] + train_inst['dims'][0] / 2
    train_inst['ymin'] = new_pos[1] - train_inst['dims'][1] / 2
    train_inst['ymax'] = new_pos[1] + train_inst['dims'][1] / 2

    # Update orientation according to rotation
    train_inst['new_alpha'] += np.deg2rad(angle)
    train_inst['new_alpha'] %= 2 * np.pi

    # Add random cropping
    h, w, _ = img.shape
    # Ensure new_h and new_w are not zero and within image bounds
    new_h = max(1, int(h * np.random.uniform(0.8, 1)))
    new_w = max(1, int(w * np.random.uniform(0.8, 1)))
    # Ensure start_x and start_y do not exceed image dimensions
    start_x = np.random.randint(0, max(1, w - new_w))
    start_y = np.random.randint(0, max(1, h - new_h))
    img = img[start_y:start_y + new_h, start_x:start_x + new_w]

    # Update position according to cropping
    train_inst['xmin'] = max(0, train_inst['xmin'] - start_x)
    train_inst['ymin'] = max(0, train_inst['ymin'] - start_y)
    train_inst['xmax'] = min(new_w, train_inst['xmax'] - start_x)
    train_inst['ymax'] = min(new_h, train_inst['ymax'] - start_y)

    # Flip the image
    flip = np.random.binomial(1, 0.5)
    if flip > 0.5:
        img = cv2.flip(img, 1)
        train_inst['new_alpha'] = 2.*np.pi - train_inst['new_alpha']

    # Resize the image to standard size
    img = cv2.resize(img, (NORM_H, NORM_W))
    img = img - np.array([[[103.939, 116.779, 123.68]]])

    # Fix orientation and confidence
    orientation = np.zeros((BIN, 2))
    confidence = np.zeros(BIN)

    anchors = compute_anchors(train_inst['new_alpha'])

    for anchor in anchors:
        orientation[anchor[0]] = np.array(
            [np.cos(anchor[1]), np.sin(anchor[1])])
        confidence[anchor[0]] = 1

    confidence = confidence / np.sum(confidence)

    train_inst['orient_flipped'] = orientation
    train_inst['conf_flipped'] = confidence

    # Fix orientation and confidence
    if flip > 0.5:
        return img, train_inst['dims'], train_inst['orient_flipped'], train_inst['conf_flipped']
    else:
        return img, train_inst['dims'], train_inst['orient'], train_inst['conf']


def data_gen(all_objs, batch_size):
    num_obj = len(all_objs)

    keys = list(range(num_obj))
    np.random.shuffle(keys)

    l_bound = 0
    r_bound = batch_size if batch_size < num_obj else num_obj

    while True:
        if l_bound == r_bound:
            l_bound = 0
            r_bound = batch_size if batch_size < num_obj else num_obj
            np.random.shuffle(keys)

        currt_inst = 0
        x_batch = np.zeros((r_bound - l_bound, 224, 224, 3))
        d_batch = np.zeros((r_bound - l_bound, 3))
        o_batch = np.zeros((r_bound - l_bound, BIN, 2))
        c_batch = np.zeros((r_bound - l_bound, BIN))

        for key in keys[l_bound:r_bound]:
            # augment input image and fix object's orientation and confidence
            image, dimension, orientation, confidence = prepare_input_and_output(
                all_objs[key])

            # plt.figure(figsize=(5,5))
            # plt.imshow(image/255./2.); plt.show()
            # print dimension
            # print orientation
            # print confidence

            x_batch[currt_inst, :] = image
            d_batch[currt_inst, :] = dimension
            o_batch[currt_inst, :] = orientation
            c_batch[currt_inst, :] = confidence

            currt_inst += 1

        yield x_batch, [d_batch, o_batch, c_batch]

        l_bound = r_bound
        r_bound = r_bound + batch_size
        if r_bound > num_obj:
            r_bound = num_obj


def l2_normalize(x):
    return tf.nn.l2_normalize(x, axis=2)
