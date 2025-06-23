# %% Imports

import matplotlib.pyplot as plt
from PIL import Image, ImageFilter
import numpy as np
import random
import operator
import cv2
from sklearn.model_selection import train_test_split
import tensorflow as tf

import time
from pathlib import Path
import shutil

import ctypes

import configparser

seed = 10

from tensorflow.keras import layers
from tensorflow import keras

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

import subprocess as sp
import os

import skimage.transform as st
import numpy as np
import os
import skimage.io as io
import skimage.transform as trans
import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import regularizers, optimizers

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, MaxPooling1D, Dropout, UpSampling2D, concatenate, \
    Reshape, Concatenate, BatchNormalization, DepthwiseConv2D, Layer

from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.layers import MultiHeadAttention

import os
from glob import glob
import time
import re
import argparse
import nibabel as nib
import pandas as pd
from medpy.metric.binary import hd, dc, hd95
import numpy as np

from skimage.io import imread

import pydicom

# %% Model
training = True  # flag
class StochasticDepth(layers.Layer):
    """
    Stochastic depth.
    """

    def __init__(self, drop_prop, **kwargs):
        super(StochasticDepth, self).__init__(**kwargs)
        self.drop_prob = drop_prop

    def call(self, x, training=training):
        if training:
            keep_prob = 1 - self.drop_prob
            #shape = (tf.shape(x)[0],) + (1,) * (len(tf.shape(x)) - 1)
            shape = tf.shape(x)
            random_tensor = keep_prob + tf.random.uniform(shape, 0, 1)
            random_tensor = tf.floor(random_tensor)
            return (x / keep_prob) * random_tensor
        return x

def wide_focus(x, filters, dropout_rate):
    x1 = layers.Conv2D(filters, 3, padding='same', activation=tf.nn.gelu)(x)
    x1 = layers.Dropout(0.1)(x1)
    x2 = layers.Conv2D(filters, 3, padding='same', dilation_rate=2, activation=tf.nn.gelu)(x)
    x2 = layers.Dropout(0.1)(x2)
    added = layers.Add()([x1, x2])
    x_out = layers.Conv2D(filters, 3, padding='same', activation=tf.nn.gelu)(added)
    x_out = layers.Dropout(0.1)(x_out)

    return x_out

class Attention(Layer):
    """
    Convolutional Attention module
    """

    def __init__(self,
                     dim_out,
                     num_heads,
                     proj_drop=0.0,
                     kernel_size=3,
                     stride_kv=1,
                     stride_q=1,
                     padding_kv="same",
                     padding_q="same",
                     attention_bias=True):
        super().__init__()
        self.stride_kv = stride_kv
        self.stride_q = stride_q
        self.dim = dim_out
        self.num_heads = num_heads

        self.conv_proj_q = self._build_projection(kernel_size, stride_q, padding_q)
        self.conv_proj_k = self._build_projection(kernel_size, stride_kv, padding_kv)
        self.conv_proj_v = self._build_projection(kernel_size, stride_kv, padding_kv)

        self.attention = MultiHeadAttention(self.num_heads, dim_out, use_bias=attention_bias)
        self.proj_drop = Dropout(proj_drop)

    @staticmethod
    def _build_projection(kernel_size, stride, padding):
        proj = Sequential([
            DepthwiseConv2D(kernel_size, padding=padding, strides=stride, use_bias=False),
            layers.LayerNormalization(), ])
        return proj

    def call_conv(self, x, h, w):
        q = self.conv_proj_q(x)
        k = self.conv_proj_k(x)
        v = self.conv_proj_v(x)

        return q, k, v

    def call(self, inputs, mask=None, training=training, h=1, w=1):
        x = inputs
        q, k, v = self.call_conv(x, h, w)
        x = self.attention(q, v, key=k)
        if training:
            x = self.proj_drop(x)

        return x

def att(x_in,
        num_heads,
        dpr,
        proj_drop=0.0,
        attention_bias=True,
        padding_q="same",
        padding_kv="same",
        stride_kv=2,
        stride_q=1):
    """
    Convolutional Attention module & Wide-Focus module together
    """

    b, h, w, c = x_in.shape

    attention_output = Attention(
            dim_out=c,
            num_heads=num_heads,
            proj_drop=proj_drop,
            attention_bias=attention_bias,
            padding_q=padding_q,
            padding_kv=padding_kv,
            stride_kv=stride_kv,
            stride_q=stride_q,
        )(x_in, h=h, w=w, training=training, mask=None)

    attention_output = StochasticDepth(dpr)(attention_output)
    attention_output = Conv2D(x_in.shape[-1], 3, 1, padding="same", activation="relu")(attention_output)
    x2 = layers.Add()([attention_output, x_in])
    x3 = layers.LayerNormalization(epsilon=1e-5)(x2)
    x3 = wide_focus(x3, filters=c, dropout_rate=0.0)
    x3 = StochasticDepth(dpr)(x3)
    x3 = layers.Add()([x3, x2])

    return x3


image_size = None
input_shape = None
def create_model(
            image_size=image_size,
            input_shape=input_shape,
    ):

    inputs = layers.Input(input_shape)
    # attention heads and filters per block
    att_heads = [2, 2, 2, 2, 2, 2, 2, 2, 2]
    filters = [16, 32, 64, 128, 384, 128, 64, 32, 16]

    # number of blocks used in the model
    blocks = len(filters)

    stochastic_depth_rate = 0.0

    dpr = [x for x in np.linspace(0, stochastic_depth_rate, blocks)]

    initializer = 'he_normal'
    drp_out = 0.3
    act = "relu"

    # Multi-scale input
    scale_img_2 = layers.AveragePooling2D(2, 2)(inputs)
    scale_img_3 = layers.AveragePooling2D(2, 2)(scale_img_2)
    scale_img_4 = layers.AveragePooling2D(2, 2)(scale_img_3)

    # first block
    #x1 = layers.LayerNormalization(epsilon=1e-5)(inputs[:, :, :, -1])
    #x11 = tf.expand_dims(x1, -1)
    x11 = Conv2D(filters[0], 3, 1, padding="same", activation=act, kernel_initializer=initializer)(inputs)
    x11 = Conv2D(filters[0], 3, 1, padding="same", activation=act, kernel_initializer=initializer)(x11)
    x11 = Dropout(drp_out)(x11)
    x11 = MaxPooling2D((2, 2))(x11)
    out = att(x11, att_heads[0], dpr[0])
    skip1 = out
    print("\nBlock 1 -> input:", inputs.shape, "output:", skip1.shape)

    # second block
    x1 = layers.LayerNormalization(epsilon=1e-5)(out)
    x11 = x1
    x11 = concatenate([Conv2D(filters[0], 3, padding="same", activation=act)(scale_img_2), x11], axis=3)
    x11 = Conv2D(filters[1], 3, 1, padding="same", activation=act, kernel_initializer=initializer)(x11)
    x11 = Conv2D(filters[1], 3, 1, padding="same", activation=act, kernel_initializer=initializer)(x11)
    x11 = Dropout(drp_out)(x11)
    x11 = MaxPooling2D((2, 2))(x11)
    out = att(x11, att_heads[1], dpr[1])
    skip2 = out
    print("Block 2 -> input:", x1.shape, "output:", skip2.shape)

    # third block
    x1 = layers.LayerNormalization(epsilon=1e-5)(out)
    x11 = x1
    x11 = concatenate([Conv2D(filters[1], 3, padding="same", activation=act)(scale_img_3), x11], axis=3)
    x11 = Conv2D(filters[2], 3, 1, padding="same", activation=act, kernel_initializer=initializer)(x11)
    x11 = Conv2D(filters[2], 3, 1, padding="same", activation=act, kernel_initializer=initializer)(x11)
    x11 = Dropout(drp_out)(x11)
    x11 = MaxPooling2D((2, 2))(x11)
    out = att(x11, att_heads[2], dpr[2])
    skip3 = out
    print("Block 3 -> input:", x1.shape, "output:", skip3.shape)

    # fourth block
    x1 = layers.LayerNormalization(epsilon=1e-5)(out)
    x11 = x1
    x11 = concatenate([Conv2D(filters[2], 3, padding="same", activation=act)(scale_img_4), x11], axis=3)
    x11 = Conv2D(filters[3], 3, 1, padding="same", activation=act, kernel_initializer=initializer)(x11)
    x11 = Conv2D(filters[3], 3, 1, padding="same", activation=act, kernel_initializer=initializer)(x11)
    x11 = Dropout(drp_out)(x11)
    x11 = MaxPooling2D((2, 2))(x11)
    out = att(x11, att_heads[3], dpr[3])
    skip4 = out
    print("Block 4 -> input:", x1.shape, "output:", skip4.shape)

    # fifth block
    x1 = layers.LayerNormalization(epsilon=1e-5)(out)
    x11 = x1
    x11 = Conv2D(filters[4], 3, 1, padding="same", activation=act, kernel_initializer=initializer)(x11)
    x11 = Conv2D(filters[4], 3, 1, padding="same", activation=act, kernel_initializer=initializer)(x11)
    x11 = Dropout(drp_out)(x11)
    x11 = MaxPooling2D((2, 2))(x11)
    out = att(x11, att_heads[4], dpr[4])
    print("Block 5 -> input:", x1.shape, "output:", out.shape)

    # sixth block
    x1 = layers.LayerNormalization(epsilon=1e-5)(out)
    x11 = x1
    x11 = Conv2D(filters[5], 2, padding="same", activation=act, kernel_initializer=initializer)(
            UpSampling2D(size=(2, 2))(x11))
    x11 = concatenate([skip4, x11], axis=3)
    x11 = Conv2D(filters[5], 3, 1, padding="same", activation=act, kernel_initializer=initializer)(x11)
    x11 = Conv2D(filters[5], 3, 1, padding="same", activation=act, kernel_initializer=initializer)(x11)
    x11 = Dropout(drp_out)(x11)
    out = att(x11, att_heads[5], dpr[5])
    print("Block 6 -> input:", x1.shape, "output:", out.shape)

    # seventh block
    x1 = layers.LayerNormalization(epsilon=1e-5)(out)
    x11 = x1
    x11 = Conv2D(filters[6], 2, padding="same", activation=act, kernel_initializer=initializer)(
            UpSampling2D(size=(2, 2))(x11))
    x11 = concatenate([skip3, x11], axis=3)
    x11 = Conv2D(filters[6], 3, 1, padding="same", activation=act, kernel_initializer=initializer)(x11)
    x11 = Conv2D(filters[6], 3, 1, padding="same", activation=act, kernel_initializer=initializer)(x11)
    x11 = Dropout(drp_out)(x11)
    out = att(x11, att_heads[6], dpr[6])
    skip7 = out
    print("Block 7 -> input:", x1.shape, "output:", skip7.shape)

    # eighth block
    x1 = layers.LayerNormalization(epsilon=1e-5)(out)
    x11 = x1
    x11 = Conv2D(filters[7], 2, padding="same", activation=act, kernel_initializer=initializer)(
            UpSampling2D(size=(2, 2))(x11))
    x11 = concatenate([skip2, x11], axis=3)
    x11 = Conv2D(filters[7], 3, 1, padding="same", activation=act, kernel_initializer=initializer)(x11)
    x11 = Conv2D(filters[7], 3, 1, padding="same", activation=act, kernel_initializer=initializer)(x11)
    x11 = Dropout(drp_out)(x11)
    out = att(x11, att_heads[7], dpr[7])
    skip8 = out
    print("Block 8 -> input:", x1.shape, "output:", skip8.shape)

    # nineth block
    x1 = layers.LayerNormalization(epsilon=1e-5)(out)
    x11 = x1
    x11 = Conv2D(filters[8], 2, padding="same", activation=act, kernel_initializer=initializer)(
            UpSampling2D(size=(2, 2))(x11))
    x11 = concatenate([skip1, x11], axis=3)
    x11 = Conv2D(filters[8], 3, 1, padding="same", activation=act, kernel_initializer=initializer)(x11)
    x11 = Conv2D(filters[8], 3, 1, padding="same", activation=act, kernel_initializer=initializer)(x11)
    x11 = Dropout(drp_out)(x11)
    out = att(x11, att_heads[8], dpr[8])
    skip9 = out
    print("Block 9 -> input:", x1.shape, "output:", skip9.shape)

    # Deep supervision
    #skip7 = layers.LayerNormalization(epsilon=1e-5)(UpSampling2D(size=(2, 2))(skip7))
    #out7 = Conv2D(filters[6], 3, padding="same", activation=act, kernel_initializer=initializer)(skip7)
    #out7 = Conv2D(filters[6], 3, padding="same", activation=act, kernel_initializer=initializer)(out7)
    #
    #skip8 = layers.LayerNormalization(epsilon=1e-5)(UpSampling2D(size=(2, 2))(skip8))
    #out8 = Conv2D(filters[7], 3, padding="same", activation=act, kernel_initializer=initializer)(skip8)
    #out8 = Conv2D(filters[7], 3, padding="same", activation=act, kernel_initializer=initializer)(out8)
    #
    #skip9 = layers.LayerNormalization(epsilon=1e-5)(UpSampling2D(size=(2, 2))(skip9))
    #out9 = Conv2D(filters[8], 3, padding="same", activation=act, kernel_initializer=initializer)(skip9)
    #out9 = Conv2D(filters[8], 3, padding="same", activation=act, kernel_initializer=initializer)(out9)
    #
    # ACDC
    #out7 = Conv2D(4, (1, 1), activation="sigmoid", name='pred1')(out7)
    #out8 = Conv2D(4, (1, 1), activation="sigmoid", name='pred2')(out8)
    #out9 = Conv2D(4, (1, 1), activation="sigmoid", name='final')(out9)
    #
    skip9 = layers.LayerNormalization(epsilon=1e-5)(UpSampling2D(size=(2, 2))(skip9))
    out9 = layers.Conv2D(filters[8], 3, 1, padding="same", activation=act, kernel_initializer=initializer)(skip9)  # Optional: additional conv layer
    out9 = layers.Conv2D(1, 1, activation="sigmoid")(out9)  # Reduce channels to 1 and apply sigmoid activation

    print("\n")
    #print("DS 1 -> input:", skip7.shape, "output:", out7.shape)
    #print("DS 2 -> input:", skip8.shape, "output:", out8.shape)
    print("Output -> input:", skip9.shape, "output:", out9.shape)

    #model = keras.Model(inputs=inputs, outputs=[out7, out8, out9])
    model = keras.Model(inputs=inputs, outputs=out9)
    return model

def FCT(data):
    image_size = data.shape[1]
    input_shape = (data.shape[1], data.shape[2], data.shape[3])
    return create_model(input_shape=input_shape, image_size=image_size)


# %% Settings/parameters

# ---- images

n_classes = 3 + 1

dropout_rate = 0


# %% functions

# Metrics Calculation Functions
def dc(seg, gt):
    intersection = np.sum(seg * gt)
    return (2. * intersection) / (np.sum(seg) + np.sum(gt))

def metrics_binary(img_gt, img_pred, voxel_size):
    if img_gt.ndim != img_pred.ndim:
        raise ValueError("The arrays 'img_gt' and 'img_pred' should have the same dimension, {} against {}".format(img_gt.ndim, img_pred.ndim))

    img_gt = np.clip(img_gt, 0, 1)
    img_pred = np.clip(img_pred, 0, 1)
    
    dice = dc(img_pred, img_gt)
    
    return dice

def metrics(img_gt, img_pred, voxel_size, dset="acdc"):
    """
    Function to compute the metrics between two segmentation maps given as input.
    Parameters
    ----------
    img_gt: np.array
    Array of the ground truth segmentation map.
    img_pred: np.array
    Array of the predicted segmentation map.
    voxel_size: list, tuple or np.array
    The size of a voxel of the images used to compute the volumes.
    Return
    ------
    A list of metrics in this order, [Dice LV, Volume LV, Err LV(ml),
    Dice RV, Volume RV, Err RV(ml), Dice MYO, Volume MYO, Err MYO(ml)]
    """

    # Dice = (2xIntersection)/(Union+Intersection)
    # F1 score
    # https://www.youtube.com/watch?v=AZr64OxshLo

    if img_gt.ndim != img_pred.ndim:
        raise ValueError("The arrays 'img_gt' and 'img_pred' should have the "
                         "same dimension, {} against {}".format(img_gt.ndim,
                                                                img_pred.ndim))
    if dset == "acdc":
        res = []
        # Loop on each classes of the input images
        for c in [3, 1, 2]:
            # Copy the gt image to not alterate the input
            gt_c_i = np.copy(img_gt)
            gt_c_i[gt_c_i != c] = 0

            # Copy the pred image to not alterate the input
            pred_c_i = np.copy(img_pred)
            pred_c_i[pred_c_i != c] = 0

            # Clip the value to compute the volumes
            gt_c_i = np.clip(gt_c_i, 0, 1)
            pred_c_i = np.clip(pred_c_i, 0, 1)

            # Compute the Dice
            dice = dc(gt_c_i, pred_c_i)

            # Compute volume
            volpred = pred_c_i.sum() * np.prod(voxel_size) / 1000.
            volgt = gt_c_i.sum() * np.prod(voxel_size) / 1000.

            res += [dice, volpred, volpred - volgt]

        if voxel_size == 0:
            res = [res[0], res[3], res[6]]

    elif dset == "synapse":
        res = []
        # Loop on each classes of the input images
        for c in [1, 2, 3, 4, 5, 6, 7, 8]:
            # Copy the gt image to not alterate the input
            gt_c_i = np.copy(img_gt)
            gt_c_i[gt_c_i != c] = 0

            # Copy the pred image to not alterate the input
            pred_c_i = np.copy(img_pred)
            pred_c_i[pred_c_i != c] = 0

            # Clip the value to compute the volumes
            gt_c_i = np.clip(gt_c_i, 0, 1)
            pred_c_i = np.clip(pred_c_i, 0, 1)

            # Compute the Dice
            dice = dc(gt_c_i, pred_c_i)

            res += [dice]

    return res


from tensorflow.keras import backend as K


class WarmUpLearningRateScheduler(keras.callbacks.Callback):
    """Warmup learning rate scheduler
    """

    # https://www.dlology.com/blog/bag-of-tricks-for-image-classification-with-convolutional-neural-networks-in-keras/

    def __init__(self, warmup_batches, init_lr, verbose=0):
        """Constructor for warmup learning rate scheduler
        Arguments:
            warmup_batches {int} -- Number of batch for warmup.
            init_lr {float} -- Learning rate after warmup.
        Keyword Arguments:
            verbose {int} -- 0: quiet, 1: update messages. (default: {0})
        """

        super(WarmUpLearningRateScheduler, self).__init__()
        self.warmup_batches = warmup_batches
        self.init_lr = init_lr
        self.verbose = verbose
        self.batch_count = 0
        self.learning_rates = []

    def on_batch_end(self, batch, logs=None):
        self.batch_count = self.batch_count + 1
        lr = K.get_value(self.model.optimizer.learning_rate)
        self.learning_rates.append(lr)

    def on_batch_begin(self, batch, logs=None):
        if self.batch_count <= self.warmup_batches:
            lr = self.batch_count * self.init_lr / self.warmup_batches
            #print(f"Setting learning rate to: {lr}")
            #K.set_value(self.model.optimizer.learning_rate, lr)
            print("Setting learning rate to: {}".format(lr))
            self.model.optimizer.learning_rate = lr
            if self.verbose > 0:
                print('\nBatch %05d: WarmUpLearningRateScheduler setting learning '
                      'rate to %s.' % (self.batch_count + 1, lr))

###############################################################################

def sep_gen(data, ismask, seed=seed, batch_size=15, dset="training"):
    if dset == "training":
        if ismask:
            datagen = tf.keras.preprocessing.image.ImageDataGenerator(
                rotation_range=10, # original value: 360
                zoom_range=.2,
                shear_range=.1,
                # fill_mode="reflect",
                width_shift_range=.3,
                height_shift_range=.3,
                horizontal_flip=True,
                vertical_flip=True,
                preprocessing_function=lambda x: np.where(x > 0, 1, 0).astype(x.dtype),
            )
        else:
            datagen = tf.keras.preprocessing.image.ImageDataGenerator(
                rotation_range=10, # original value: 360
                zoom_range=.2,
                shear_range=.1,
                width_shift_range=.3,
                height_shift_range=.3,
                horizontal_flip=True,
                vertical_flip=True,
            )
    elif dset == "validation":
        if ismask:
            datagen = tf.keras.preprocessing.image.ImageDataGenerator(
                preprocessing_function=lambda x: np.where(x > 0, 1, 0).astype(x.dtype)
            )

        else:
            datagen = tf.keras.preprocessing.image.ImageDataGenerator()
    else:
        raise ValueError("The argument \"dset\" can either be \"training\" or \"validation\".")

    num_samples, height, width, channels = data.shape
    data = data.reshape((num_samples, height, width, channels))

    return datagen.flow(data, batch_size=batch_size, seed=seed)


def unite_gen(X, y, batch_size, dset):
    gen_X = sep_gen(X, False, batch_size=batch_size, dset=dset)
    gen_y = sep_gen(y, True, batch_size=batch_size, dset=dset)

    while True:
        X_batch = next(gen_X)
        y_batch = next(gen_y)

        # Reshape y_batch to have the correct output shape
        y_batch = y_batch.reshape((-1, y.shape[1], y.shape[2], 1))  # Assuming (num_samples, height, width, 1)

        yield (X_batch, y_batch)


# %% Data preparation

def resize_image(image, target_shape):
    return cv2.resize(image, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_NEAREST)

def normalize_T1(image):
    image[image < 0] = 0
    upper = 960.4 + (142.8 + 311)
    lower = 960.4 - (142.8 + 311)
    image = np.clip(image, lower, upper)
    image = (image - lower) / (upper - lower)
    return image
    
def load_dicom_files(folder_path, target_shape=(384, 384)):
    manifest_path = os.path.join(folder_path, 'manifest.csv')
    if not os.path.exists(manifest_path):
        raise ValueError(f"Manifest file does not exist: {manifest_path}")
    manifest = pd.read_csv(manifest_path)

    images = []

    for f in sorted(os.listdir(folder_path)):
        #print(f)
        if f.endswith(".dcm"):
            dicom_files = os.path.join(folder_path, f)
            image = pydicom.dcmread(dicom_files).pixel_array
            #print("original shape:",image.shape)
            #print(image[180:190, 180])
            image = resize_image(image, target_shape)
            #print("shape after resize:",image.shape)

            rows = manifest[manifest['filename'] == f]
            #print("manifest(filename):",manifest['filename'])
            #print("f:",f)
            #print(rows)
            if not rows.empty:
                for _, row in rows.iterrows():
                    series_name = row['series discription']
                    if 'T1MAP' in series_name:
                        image = normalize_T1(image)
                        #print("T1:",image[180:190, 180])
                    else:
                        image = normalise_input(image, opt='quantile', per_channel=True)
                        #print("Not T1:",image[180:190, 180])
            else:
                image = normalise_input(image, opt='quantile', per_channel=True)

            images.append(image)

    return np.stack(images, axis=-1)


def load_mask(mask_path, target_shape=(384, 384)):
    mask = imread(mask_path, as_gray=True)
    if mask.shape != target_shape:
        mask = resize_image(mask, target_shape)
    return mask


def load_data(data_folder, target_shape=(384, 384)):
    participants = [os.path.join(data_folder, p) for p in os.listdir(data_folder) if
                    os.path.isdir(os.path.join(data_folder, p))]
    data = []
    masks = []
    folder_names = []


    for participant in participants:
        try:
            dicom_images = load_dicom_files(participant, target_shape)
            mask_path = os.path.join(participant, '1_mask.png')
            #if not os.path.exists(mask_path):
             #   raise ValueError(f"Mask file not found for participant: {participant}")
            mask = load_mask(mask_path, target_shape)

            # Ensure consistent shapes for masks and images
            if dicom_images.shape[:2] != mask.shape:
                raise ValueError(f"Shape mismatch between images and mask for participant: {participant}")
  
            data.append(dicom_images)
            masks.append(mask)
            folder_names.append(os.path.basename(participant))  # Get folder name

        except ValueError as e:
            print(f"Error processing {participant}: {e}")
    if not data or not masks:
        raise ValueError("No valid DICOM images or masks found in the dataset.")
        
    print(f"Shape of dicom_images for participant {participant}: {dicom_images.shape}")

    data = np.array(data)
    masks = np.array(masks)

    masks = np.expand_dims(masks, axis=-1)

    return data, masks, folder_names

def load_test_data(data_folder, target_shape=(384, 384)):
    participants = [os.path.join(data_folder, p) for p in os.listdir(data_folder) if
                    os.path.isdir(os.path.join(data_folder, p))]
    data = []
    folder_names = []

    for participant in participants:
        try:
            dicom_images = load_dicom_files(participant, target_shape)
            data.append(dicom_images)
            folder_names.append(os.path.basename(participant))  # Get folder name

        except ValueError as e:
            print(f"Error processing {participant}: {e}")
    
    if not data:
        raise ValueError("No valid DICOM images found in the dataset.")

    data = np.array(data)

    return data, folder_names


#def split_data(data, masks, train_size = 0.8, val_size = 0.2, random_state = None):
#    data_train, data_val, masks_train, masks_val = train_test_split(data, masks, test_size=val_size, random_state=random_state)
#    return data_train, data_val, masks_train, masks_val

def split_data(data, masks, folder_names, train_size=0.8, val_size=0.2, random_state=None):
    data_train, data_val, masks_train, masks_val, folders_train, folders_val = train_test_split(
        data, masks, folder_names, test_size=val_size, random_state=random_state)
    return data_train, data_val, masks_train, masks_val, folders_train, folders_val

def normalise_input(Imgs, opt='hist', per_channel=True):
    Imgs = Imgs.astype(np.float32)
    if opt == 'std':
       # to the range [0,1] being 1sd
       if len(Imgs.shape) == 4:
          for j in range(Imgs.shape[0]):
             if per_channel:
                for k in range(7):
                   std = np.std(Imgs[j,:,:,k])
                   Imgs[j,:,:,k] = Imgs[j,:,:,k]/(std+0.000000001)
                else:
                   print('Error, function not implemented yet, please fix')

       elif len(Imgs.shape) == 3:
          if per_channel:
             for k in range(7):
                std = np.std(Imgs[:, :, k])
                Imgs[:, :, k] = Imgs[:, :, k] / (std + 0.000000001)
             else:
                print('Error, function not implemented yet, please fix')
       else:
          Imgs = None
          print('error, shape not recognised')

    # 20190806 preferred
    elif opt =='quantile':
       if len(Imgs.shape) == 4:
          for j in range(Imgs.shape[0]):
             if per_channel:
                for k in range(Imgs.shape[-1]):
                   # Adaptive Equalization
                   Imgs[j,:,:,k] = (Imgs[j, :, :, k] - np.quantile(Imgs[j, :, :, k], 0.01)) / (np.quantile(Imgs[j, :, :, k], 0.99) - np.quantile(Imgs[j, :, :, k], 0.01) + 1e-5)
                   Imgs[j,:,:,k] = np.clip(Imgs[j,:,:,k], 0, 1)
             else:

                Imgs[j] = (Imgs[j] - np.quantile(Imgs[j], 0.01)) / (np.quantile(Imgs[j], 0.99) - np.quantile(Imgs[j], 0.01) + 1e-5)
                Imgs[j] = np.clip(Imgs[j], 0, 1)
       elif len(Imgs.shape) == 3:
          if per_channel:
             for k in range(Imgs.shape[-1]):
                # Adaptive Equalization
                img1 = (Imgs[:, :, k] - np.quantile(Imgs[:, :, k], 0.01)) / (np.quantile(Imgs[:, :, k], 0.99) - np.quantile(Imgs[:, :, k], 0.01) + 1e-5)
                img1 = np.clip(img1, 0, 1)
                Imgs[:, :, k] = img1
          else:
             Imgs = (Imgs - np.quantile(Imgs, 0.01)) / (np.quantile(Imgs, 0.99) - np.quantile(Imgs, 0.01) + 1e-5)
             Imgs = np.clip(Imgs, 0, 1)

       elif len(Imgs.shape) == 2:
          img1 = (Imgs - np.quantile(Imgs, 0.01)) / (np.quantile(Imgs, 0.99) - np.quantile(Imgs, 0.01) + 1e-5)
          img1 = np.clip(img1, 0, 1)
          Imgs = img1

       else:
          Imgs = None
          print('error, shape not recognised')

    elif opt == 'hist':
       # to the range [0,1] being 1sd
       if len(Imgs.shape) == 4:
          for img in Imgs:
             for k in range(Imgs.shape[-1]):
                # Adaptive Equalization
                img1 = (img[:, :, k] - np.quantile(img[:, :, k], 0.01)) / (np.quantile(img[:, :, k], 0.99) - np.quantile(img[:, :, k], 0.01) + 1e-5) * 2 - 1
                img1 = np.clip(img1, -1, 1)
                img_adapteq = exposure.equalize_hist(img1)
                img[:, :, k] = img_adapteq
       elif len(Imgs.shape) == 3:
          for k in range(Imgs.shape[-1]):
             # Adaptive Equalization
             img1 = (Imgs[:, :, k] - np.quantile(Imgs[:, :, k], 0.01)) / (np.quantile(Imgs[:, :, k], 0.99) - np.quantile(Imgs[:, :, k], 0.01) + 1e-5) * 2 - 1
             img1 = np.clip(img1, -1, 1)
             img_adapteq = exposure.equalize_hist(img1)
             Imgs[:, :, k] = img_adapteq

       elif len(Imgs.shape) == 2:
          img1 = (Imgs - np.quantile(Imgs, 0.01)) / (np.quantile(Imgs, 0.99) - np.quantile(Imgs, 0.01) + 1e-5) * 2 - 1
          img1 = np.clip(img1, -1, 1)
          img_adapteq = exposure.equalize_hist(img1)
          Imgs = img_adapteq
       else:
          Imgs = None
          print('error, shape not recognised')

    else:
       Imgs = None

    return Imgs

def save_predictions(val_data, val_masks, predicted_classes, save_dir, val_folders):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    num_samples = val_data.shape[0]

    for i in range(num_samples):
        folder_name = val_folders[i]
        folder_path = os.path.join(save_dir, folder_name)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        pred_mask_path = os.path.join(folder_path, 'pred_mask.png')

        # Save the predicted mask
        cv2.imwrite(pred_mask_path, (predicted_classes[i, :, :, 0] * 255).astype(np.uint8))