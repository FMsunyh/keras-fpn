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
# @Time    : 11/8/2018 4:54 PM
# @Author  : Firmin.Sun (fmsunyh@gmail.com)
# @Software: ZJ_AI
# -----------------------------------------------------
# -*- coding: utf-8 -*-

import keras
import tensorflow

def crop_and_resize(image, boxes, size):
    '''
    Crop the image given boxes and resize with bilinear interplotation.
    :param image: Input image of shape (1, image_height, image_width, depth)
    :param boxes: Regions of interest of shape (num_boxes, 4),
    :param size: Fixed size [h, w], e.g. [7, 7], for the output slices.
    :return: 4D Tensor (number of regions, slice_height, slice_width, channels)
    '''

    box_ind = keras.backend.zeros_like(boxes, "int32")
    box_ind = box_ind[:, 0]
    box_ind = keras.backend.reshape(box_ind, [-1])

    boxes = keras.backend.cast(keras.backend.reshape(boxes, [-1, 4]),'float32')

    return tensorflow.image.crop_and_resize(image, boxes, box_ind, size)
