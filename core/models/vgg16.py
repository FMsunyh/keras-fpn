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
# -*- coding: utf-8 -*-
# @Time    : 10/29/2018 5:15 PM
# @Author  : sunyonghai
# @Software: ZJ_AI
# -----------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import core
import keras
from core.models.fastrcnn import FastRCNN
from core.models.region_proposal_network import RegionProposalNetwork

def VGG16RPN(inputs, training=True, num_classes=2, weights='imagenet', *args, **kwargs):
    if training:
        image, gt_boxes = inputs
    else:
        image = inputs

    image_shape = core.layers.Dimensions()(image)

    # 1.backbone
    vgg16 = keras.applications.VGG16(input_tensor=image, include_top=False, weights='imagenet')
    features_map = vgg16.get_layer(index=-2).output

    # 2. Anchors, target anchors
    # compute labels and bbox_reg_targs
    anchors = core.layers.Anchors(stride=16, base_size=16, name='generator_anchors')(features_map)
    if training:
        labels, regression_target = core.layers.AnchorTarget(name='anchor_target_generator')([anchors, image_shape, gt_boxes])

    # 3. RPN output_scores(classification), output_deltas(regression)
    rpn_input = [labels,regression_target] if training else image
    cls_loss, reg_loss, classification, regression = RegionProposalNetwork(backbone=vgg16, training=training, num_classes=2)(inputs=rpn_input, *args, **kwargs)

    model = keras.models.Model(inputs=inputs, output=[cls_loss, reg_loss], *args, **kwargs)
    return model

def VGG16FasterRCNN(inputs, training=True, num_classes=21, name='VGG16FasterRCNN', *args, **kwargs):
    if training:
        image, gt_boxes = inputs
    else:
        image = inputs

    image_shape = core.layers.Dimensions()(image)

    # 1.backbone
    vgg16 = keras.applications.VGG16(input_tensor=image, include_top=False, weights='imagenet')
    for index, layer in enumerate(vgg16.layers):
        layer.trainable = False
        if index > 5:
            break

    features_map = vgg16.get_layer(index=-2).output

    # 2. Anchors, target anchors
    # compute labels and bbox_reg_targs
    anchors = core.layers.Anchors(stride=16, base_size=16, name='generator_anchors')(features_map)
    if training:
        labels, regression_target = core.layers.AnchorTarget(name='anchor_target_generator')([anchors, image_shape, gt_boxes])

    # 3. RPN output_scores(classification), output_deltas(regression)
    rpn_input = [labels,regression_target] if training else image
    rpn_cls_loss, rpn_reg_loss, classification, regression = RegionProposalNetwork(backbone=vgg16, training=training, num_classes=2)(inputs=rpn_input, *args, **kwargs)

    # 4 Proposal (ROIs)
    proposal_boxes = core.layers.Proposal(training=training, name='proposal')(inputs=[classification, regression, anchors, image_shape])

    if training:
        proposal_boxes, proposal_labels, proposal_regression_target = core.layers.ProposalTarget(name='proposal_target',num_classes=num_classes)(inputs=[proposal_boxes, gt_boxes])

    # 3.ROI POOling
    roi_pooling_out = core.layers.RoiPooling(num_rois=128, name='roipooling')(inputs=[image_shape, features_map, proposal_boxes])

    # 4.Fast RCNN
    rcnn_input = [roi_pooling_out, proposal_labels, proposal_regression_target] if training else roi_pooling_out
    cls_loss, reg_loss, classification, regression = FastRCNN(training=training, num_classes=num_classes)(inputs= rcnn_input)

    model_output = [rpn_cls_loss, rpn_reg_loss,cls_loss, reg_loss] if training else [classification, regression, proposal_boxes]
    model = keras.models.Model(inputs=inputs, outputs=model_output, name=name)
    return model


def VGG16FasterRCNN_bbox(num_classes=21, name='VGG16FasterRCNN_bbox'):
    image = keras.layers.Input((None, None, 3))
    model = VGG16FasterRCNN(image, training=False, num_classes=num_classes, name=name)
    return model
