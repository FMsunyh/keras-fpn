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
# @Time    : 11/23/2018 10:38 AM
# @Author  : Firmin.Sun (fmsunyh@gmail.com)
# @Software: ZJ_AI
# -----------------------------------------------------
# -*- coding: utf-8 -*-
import core
import keras

class FastRCNN(object):
    def __init__(self, training=True, num_classes=21):
        '''
        Fast RCNN introduced in Faster R-CNN.
        '''

        self.training = training
        self.num_classes = num_classes

        self.network()

    def network(self):
        self.out_0 = keras.layers.TimeDistributed(keras.layers.Flatten(name='flatten'))
        self.out_1 = keras.layers.TimeDistributed(keras.layers.Dense(units=4096, activation="relu", name="fc1"))
        self.out_2 = keras.layers.TimeDistributed(keras.layers.Dropout(0.5))
        self.out_3 = keras.layers.TimeDistributed(keras.layers.Dense(units=4096, activation="relu", name="fc2"))
        self.out_4 = keras.layers.TimeDistributed(keras.layers.Dropout(0.5))
        self.classification = keras.layers.TimeDistributed(
            keras.layers.Dense(self.num_classes, activation='softmax', kernel_initializer='zero'),
            name='dense_class_{}'.format(self.num_classes))

        # note: no regression target for bg class
        self.regression = keras.layers.TimeDistributed(
            keras.layers.Dense(4 * self.num_classes, activation='linear', kernel_initializer='zero'),
            name='dense_regress_{}'.format(self.num_classes))

        if self.training:
            self.rcnn_loss = core.layers.RCNNLoss(self.num_classes, name='rcnnloss')

    def __call__(self, inputs, *args, **kwargs):
        if self.training:
            roi_pooling_out, labels, regression_target = inputs
        else:
            roi_pooling_out = inputs
        classification = None
        regression = None
        cls_loss, reg_loss = None ,None

        x = self.out_0(roi_pooling_out)
        x = self.out_1(x)
        x = self.out_2(x)
        x = self.out_3(x)
        x = self.out_4(x)
        classification = self.classification(x)
        regression = self.regression(x)

        if self.training:
            cls_loss, reg_loss = self.rcnn_loss([classification, labels, regression, regression_target])

        return cls_loss, reg_loss, classification, regression
