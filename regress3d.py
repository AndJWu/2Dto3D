import os

import cv2
import numpy as np

from network import model
from parameters import (BATCH_SIZE, BIN, MAX_JIT, NORM_H, NORM_W,
                        OVERLAP, VEHICLES, image_dir, label_dir)
from preprocessdata import dims_avg

model.load_weights('weights.hdf5')
image_dir = 'E:/workspace/2dto3d/data/2011_09_26_drive_0014_sync/2011_09_26/2011_09_26_drive_0014_sync/image_00/data/'
box2d_loc = 'E:/workspace/2dto3d/data/2011_09_26_drive_0014_sync/2011_09_26/2011_09_26_drive_0014_sync/box_2d/'
box3d_loc = 'E:/workspace/2dto3d/data/2011_09_26_drive_0014_sync/2011_09_26/2011_09_26_drive_0014_sync/box_3d/'

all_image = sorted(os.listdir(image_dir))
# np.random.shuffle(all_image)

for f in all_image:
    image_file = image_dir + f
    box2d_file = box2d_loc + f.replace('png', 'txt')
    box3d_file = box3d_loc + f.replace('png', 'txt')

    with open(box3d_file, 'w') as box3d:
        img = cv2.imread(image_file)
        img = img.astype(np.float32, copy=False)
        open(box2d_file, 'w')
        for line in open(box2d_file):
            line = line.strip().split(' ')
            truncated = np.abs(float(line[1]))
            occluded = np.abs(float(line[2]))

            obj = {'xmin': int(float(line[4])),
                   'ymin': int(float(line[5])),
                   'xmax': int(float(line[6])),
                   'ymax': int(float(line[7])),
                   }

            patch = img[obj['ymin']:obj['ymax'], obj['xmin']:obj['xmax']]
            patch = cv2.resize(patch, (NORM_H, NORM_W))
            patch = patch - np.array([[[103.939, 116.779, 123.68]]])
            patch = np.expand_dims(patch, 0)

            prediction = model.predict(patch)

            # Transform regressed angle
            max_anc = np.argmax(prediction[2][0])
            anchors = prediction[1][0][max_anc]

            if anchors[1] > 0:
                angle_offset = np.arccos(anchors[0])
            else:
                angle_offset = -np.arccos(anchors[0])

            wedge = 2.*np.pi/BIN
            angle_offset = angle_offset + max_anc*wedge
            angle_offset = angle_offset % (2.*np.pi)

            angle_offset = angle_offset - np.pi/2
            if angle_offset > np.pi:
                angle_offset = angle_offset - (2.*np.pi)

            line[3] = str(angle_offset)

            # Transform regressed dimension
            dims = dims_avg['Car'] + prediction[0][0]

            line = line + list(dims)

            # Write regressed 3D dim and oritent to file
            line = ' '.join([str(item) for item in line]) + '/n'
            box3d.write(line)

            cv2.rectangle(img, (obj['xmin'], obj['ymin']),
                          (obj['xmax'], obj['ymax']), (255, 0, 0), 3)

    # plt.figure(figsize=(10,10))
    # plt.imshow(img/255.)
    # plt.show()
