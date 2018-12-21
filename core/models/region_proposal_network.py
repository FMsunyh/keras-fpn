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
# @Time    : 11/14/2018 5:05 PM
# @Author  : Firmin.Sun (fmsunyh@gmail.com)
# @Software: ZJ_AI
# -----------------------------------------------------
# -*- coding: utf-8 -*-
import core
import keras
import numpy as np

# class RegionProposalNetwork(object):
class RegionProposalNetwork(object):
    def __init__(self, backbone, training=True, num_classes=2, feature_size=512, ratios=[0.5, 1, 2], anchor_scales=[8, 16, 32], feat_stride=16,  nms=True, **proposal_generator_params):
        '''
        Region Proposal Network introduced in Faster R-CNN.

        :param feature_size:
        :param ratios:
        :param anchor_scales:
        :param feat_stride:
        :param proposal_generator_params:
        '''
        # image, gt_boxes = self.inputs

        # self.inputs = inputs
        self.backbone = backbone
        self.num_classes = num_classes
        self.ratios = ratios
        self.anchor_scales = anchor_scales
        self.feature_size = feature_size
        self.feat_stride = feat_stride
        self.num_anchors = len(ratios) * len(anchor_scales)
        self.features_map = self.backbone.get_layer(index=-2)
        self.training = training

        self.rpn_network()

    def rpn_network(self):
        self.conv = keras.layers.Conv2D(filters=self.feature_size, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='normal', name='rpn_conv1')
        self.classification_layer = keras.layers.Conv2D(filters=self.num_anchors * self.num_classes, kernel_size=(1, 1), strides=(1, 1), padding='valid', kernel_initializer='uniform', name='rpn_classification')
        self.regression_layer = keras.layers.Conv2D(filters=self.num_anchors * 4, kernel_size=(1, 1),strides=(1, 1), padding='valid',kernel_initializer='zero', name='rpn_regression')

    def __call__(self, inputs, *args, **kwargs):
        if self.training:
            labels, regression_target = inputs
        # else:
        #     image = inputs

        classification = None
        regression = None
        cls_loss, reg_loss = None , None

        x = self.conv(self.features_map.output)
        classification = self.classification_layer(x) # batch * h * w * (num_anchors * 2)
        classification = core.layers.TensorReshape((-1, self.num_classes), name='rpn_classification_reshape')(classification)
        classification = keras.layers.Activation('softmax', name='classification_softmax')(classification)

        regression = self.regression_layer(x)
        regression = core.layers.TensorReshape((-1, 4), name='rpn_boxes_reshaped')(regression)
        regression = keras.layers.Activation('linear', name='regression_linear')(regression)

        if self.training:
            cls_loss,reg_loss = core.layers.RPNLoss(num_classes=self.num_classes, name='rpn_loss')([classification, labels, regression, regression_target])
        return cls_loss, reg_loss, classification, regression
