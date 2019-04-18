#!/usr/bin/python3

"""
Copyright 2018-2019  Firmin.Sun (fmsunyh@gmail.com)

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
# -----------------------------------------------------
# @Time    : 11/28/2018 1:37 PM
# @Author  : Firmin.Sun (fmsunyh@gmail.com)
# @Software: ZJ_AI
# -----------------------------------------------------
# -*- coding: utf-8 -*-
import keras
import core
import tensorflow as tf

class RoiPooling(keras.layers.Layer):
    """ROI pooling layer for 2D inputs.
    See Spatial Pyramid Pooling in Deep Convolutional Networks for Visual Recognition,
    keras.backend. He, X. Zhang, S. Ren, J. Sun
    """

    def __init__(self, num_rois, pool_size=7, strides=1, **kwargs):
        self.pool_size = pool_size
        self.num_rois = num_rois
        self.strides = strides

        super(RoiPooling, self).__init__(**kwargs)

    def build(self, input_shape):
        self.nb_channels = input_shape[1][3]

    def compute_output_shape(self, input_shape):
        return None, None, self.pool_size, self.pool_size, self.nb_channels

    def call(self, inputs , mask=None):
        assert (len(inputs) == 3)
        image_shape, features_map, rois = inputs
        '''
           Crop the image given boxes and resize with bilinear interplotation.
           :param image: Input image of shape (None, image_height, image_width, depth)
           :param rois: Regions of interest of shape (None, num_boxes, 4),
           :return: 4D Tensor (None, number of regions, slice_height, slice_width, channels)
        '''

        # TODO: Fix usage of batch index
        rois = rois[0]

        rois = keras.backend.cast(rois, keras.backend.floatx())
        rois = rois / self.strides

        x1 = rois[..., 0]
        y1 = rois[..., 1]
        x2 = rois[..., 2]
        y2 = rois[..., 3]

        h = keras.backend.cast(image_shape[0], keras.backend.floatx())
        w = keras.backend.cast(image_shape[1], keras.backend.floatx())

        x1 /= w
        y1 /= h
        x2 /= w
        y2 /= h

        x1 = keras.backend.expand_dims(x1, axis=-1)
        y1 = keras.backend.expand_dims(y1, axis=-1)
        x2 = keras.backend.expand_dims(x2, axis=-1)
        y2 = keras.backend.expand_dims(y2, axis=-1)

        rois = keras.backend.concatenate([y1, x1, y2, x2], axis=-1)
        rois = keras.backend.reshape(rois, (-1, 4))

        # Won't be back-propagated to rois anyway, but to save time
        bboxes = tf.stop_gradient(rois)

        pool_size = self.pool_size * 2

        slices = core.backend.crop_and_resize(features_map, bboxes, (pool_size, pool_size))

        slices = keras.layers.MaxPooling2D(pool_size=(2,2), padding='same')(slices)
        out = keras.backend.expand_dims(slices, axis=0)

        return out

    def get_config(self):
        return {
            'pool_size': self.pool_size,
            "strides": self.strides
        }