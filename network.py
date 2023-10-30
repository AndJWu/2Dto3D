# Construct the network
# Use ResNet or similar more modern architecture
from keras.applications.resnet import ResNet50
from keras.layers import Dense, Input
from keras.layers.activation import LeakyReLU
from keras.layers.convolutional import (Conv2D, Convolution2D, MaxPooling2D,
                                        ZeroPadding2D)
from keras.layers.core import Dense, Dropout, Flatten, Lambda, Reshape
from keras.models import Model
from keras.regularizers import l2

from parameters import BIN
from preprocessdata import l2_normalize

# Load ResNet model, pre-trained on ImageNet; exclude top FC layer
base_model = ResNet50(weights='imagenet', include_top=False,
                      input_shape=(224, 224, 3))

# Make base_model layers non-trainable
for layer in base_model.layers:
    layer.trainable = False

# You can choose to retrain some of the higher layers
for layer in base_model.layers[-4:]:
    layer.trainable = True

# Now add your custom layers
x = base_model.output
x = Flatten()(x)

# Regularization strength
reg_strength = 0.01

dimension = Dense(512, kernel_regularizer=l2(reg_strength))(x)
dimension = LeakyReLU(alpha=0.1)(dimension)
dimension = Dropout(0.5)(dimension)
dimension = Dense(3)(dimension)
dimension = LeakyReLU(alpha=0.1, name='dimension')(dimension)

orientation = Dense(256, kernel_regularizer=l2(reg_strength))(x)
orientation = LeakyReLU(alpha=0.1)(orientation)
orientation = Dropout(0.5)(orientation)
orientation = Dense(BIN*2)(orientation)
orientation = LeakyReLU(alpha=0.1)(orientation)
orientation = Reshape((BIN, -1))(orientation)
orientation = Lambda(l2_normalize, name='orientation')(orientation)

confidence = Dense(256, kernel_regularizer=l2(reg_strength))(x)
confidence = LeakyReLU(alpha=0.1)(confidence)
confidence = Dropout(0.5)(confidence)
confidence = Dense(BIN, activation='softmax', name='confidence')(confidence)

model = Model(base_model.input, outputs=[dimension, orientation, confidence])

# model.load_weights('initial_weights.h5')
