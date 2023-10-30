import os

import numpy as np
import tensorflow as tf
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from keras.optimizers import SGD

from network import model
from preprocessdata import all_objs, data_gen

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def combined_orientation_loss(y_true, y_pred):
    anchors = tf.reduce_sum(tf.square(y_true), axis=2)
    anchors = tf.greater(anchors, tf.constant(0.5))
    anchors = tf.reduce_sum(tf.cast(anchors, tf.float32), 1)

    cos_distance = -(y_true[:, :, 0]*y_pred[:, :, 0] + y_true[:, :, 1]*y_pred[:, :, 1])
    euclidean_distance = tf.square(y_true - y_pred)
    euclidean_distance = tf.reduce_sum(euclidean_distance, axis=2)

    loss = cos_distance + euclidean_distance
    loss = tf.reduce_sum(loss, axis=1)
    loss = loss / anchors

    return tf.reduce_mean(loss)



# Early stopping to prevent overfitting
early_stop = EarlyStopping(
    monitor='val_loss',
    min_delta=0.001,
    patience=10,
    mode='min',
    verbose=1
)

# Save the best models during training
checkpoint = ModelCheckpoint(
    'weights.hdf5',
    monitor='val_loss',
    verbose=1,
    save_best_only=True,
    mode='min',
    save_freq='epoch'
)

# TensorBoard for visualizing training progress
tensorboard = TensorBoard(
    log_dir='../logs/',
    histogram_freq=0,
    write_graph=True,
    write_images=False
)

all_exams = len(all_objs)
trv_split = int(0.9*all_exams)
batch_size = 8

# Shuffle the dataset
np.random.shuffle(all_objs)

# Split data into training and validation sets
train_gen = data_gen(all_objs[:trv_split], batch_size)
valid_gen = data_gen(all_objs[trv_split:all_exams], batch_size)

# Calculate the number of batches for training and validation sets
train_num = int(np.ceil(trv_split/batch_size))
valid_num = int(np.ceil((all_exams - trv_split)/batch_size))

minimizer = SGD(learning_rate=0.0001)
model.compile(optimizer='adam',  # minimizer,
              loss={'dimension': 'mean_squared_error',
                    'orientation': combined_orientation_loss, 'confidence': 'mean_squared_error'},
              loss_weights={'dimension': 1., 'orientation': 1., 'confidence': 1.})
model.fit(train_gen, steps_per_epoch=train_num,
          epochs=5,  # 500
          verbose=1,
          validation_data=valid_gen,
          validation_steps=valid_num,
          callbacks=[early_stop, checkpoint, tensorboard],
          max_queue_size=3,
          workers=1, use_multiprocessing=False)
