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
# @Time    : 11/28/2018 1:36 PM
# @Author  : Firmin.Sun (fmsunyh@gmail.com)
# @Software: ZJ_AI
# -----------------------------------------------------
# -*- coding: utf-8 -*-
import keras
import core
import numpy as np

class Anchors(keras.layers.Layer):
    def __init__(self, base_size, stride, *args, **kwargs):
        self.base_size = base_size
        self.stride    = stride
        super(Anchors, self).__init__(*args, **kwargs)

    def call(self, inputs, **kwargs):
        '''
        Generator anchor with feature map
        :param inputs: shape [1,H,W,channels]
        :param kwargs:
        :return: anchors ,shape [1,H,W,9*4]
        '''
        features = inputs
        features_map_shape = keras.backend.shape(features)[1:3]

        # generate proposals from bbox deltas and shifted anchors
        anchors = core.backend.anchor(base_size=self.base_size)
        anchors = core.backend.shift(features_map_shape, self.stride, anchors)
        anchors = keras.backend.expand_dims(anchors, axis=0)
        return anchors

    def compute_output_shape(self, input_shape):
        if None not in input_shape[1:]:
            total = np.prod(input_shape[1:]) // 4
            return (input_shape[0], total, 4)
        else:
            return (input_shape[0], None, 4)

    def get_config(self):
        return {
            'base_size': self.base_size,
            'stride'     : self.stride,
        }
