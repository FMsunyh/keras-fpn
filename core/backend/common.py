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

import keras.backend
import tensorflow as tf

def bbox_transform_inv(boxes, deltas):
    """
    Calculate object boxes with boxes and deltas
    mostly identical to "https://github.com/tryolabs/luminoth.git"
    :param boxes: (N, 4) ndarray of float
    :param deltas: (N, 4) ndarray of float
    :return: pred_boxes: (N, 4) ndarray of float
    """
    boxes  = keras.backend.reshape(boxes, (-1, 4))
    deltas = keras.backend.reshape(deltas, (-1, 4))

    widths  = boxes[:, 2] - boxes[:, 0]
    heights = boxes[:, 3] - boxes[:, 1]
    ctr_x   = boxes[:, 0] + 0.5 * widths
    ctr_y   = boxes[:, 1] + 0.5 * heights

    dx = deltas[:, 0]
    dy = deltas[:, 1]
    dw = deltas[:, 2]
    dh = deltas[:, 3]

    pred_ctr_x = dx * widths + ctr_x
    pred_ctr_y = dy * heights + ctr_y
    pred_w     = keras.backend.exp(dw) * widths
    pred_h     = keras.backend.exp(dh) * heights

    pred_boxes_x1 = pred_ctr_x - 0.5 * pred_w
    pred_boxes_y1 = pred_ctr_y - 0.5 * pred_h
    pred_boxes_x2 = pred_ctr_x + 0.5 * pred_w
    pred_boxes_y2 = pred_ctr_y + 0.5 * pred_h

    pred_boxes = keras.backend.stack([pred_boxes_x1, pred_boxes_y1, pred_boxes_x2, pred_boxes_y2], axis=1)
    pred_boxes = keras.backend.expand_dims(pred_boxes, axis=0)

    return pred_boxes

def bbox_transform(gt_rois, ex_rois):
    """
    Compute bounding-box regression targets for an image.
    :param ex_rois: (N, 4) ndarray of float
    :param gt_rois: (N, 4) ndarray of float
    :return: targets: (N, 4) ndarray of float
    """
    ex_widths = ex_rois[:, 2] - ex_rois[:, 0] + 1.0
    ex_heights = ex_rois[:, 3] - ex_rois[:, 1] + 1.0
    ex_ctr_x = ex_rois[:, 0] + 0.5 * ex_widths
    ex_ctr_y = ex_rois[:, 1] + 0.5 * ex_heights

    gt_widths = gt_rois[:, 2] - gt_rois[:, 0] + 1.0
    gt_heights = gt_rois[:, 3] - gt_rois[:, 1] + 1.0
    gt_ctr_x = gt_rois[:, 0] + 0.5 * gt_widths
    gt_ctr_y = gt_rois[:, 1] + 0.5 * gt_heights

    targets_dx = (gt_ctr_x - ex_ctr_x) / ex_widths
    targets_dy = (gt_ctr_y - ex_ctr_y) / ex_heights
    targets_dw = keras.backend.log(gt_widths / ex_widths)
    targets_dh = keras.backend.log(gt_heights / ex_heights)

    targets = keras.backend.stack((targets_dx, targets_dy, targets_dw, targets_dh))

    targets = keras.backend.transpose(targets)

    return keras.backend.cast(targets, keras.backend.floatx())

def anchor(base_size=16, ratios=None, scales=None):
    """
    Generates a regular grid of multi-aspect and multi-scale anchor boxes.
    """
    if ratios is None:
        ratios = keras.backend.cast([0.5, 1, 2], keras.backend.floatx())

    if scales is None:
        scales = keras.backend.cast([2**3, 2**4, 2**5], keras.backend.floatx())
    base_anchor = keras.backend.cast([1, 1, base_size, base_size], keras.backend.floatx()) - 1
    base_anchor = keras.backend.expand_dims(base_anchor, 0)

    ratio_anchors = _ratio_enum(base_anchor, ratios)
    anchors = _scale_enum(ratio_anchors, scales)

    return anchors

def shift(shape, stride, anchors):
    """
    Produce shifted anchors based on shape of the map and stride size
    """
    shift_x = keras.backend.arange(0, shape[1]) * stride
    shift_y = keras.backend.arange(0, shape[0]) * stride

    shift_x, shift_y = tf.meshgrid(shift_x, shift_y)
    shift_x = keras.backend.reshape(shift_x, [-1])
    shift_y = keras.backend.reshape(shift_y, [-1])

    shifts = keras.backend.stack([
        shift_x,
        shift_y,
        shift_x,
        shift_y
    ], axis=0)

    shifts            = keras.backend.transpose(shifts)
    number_of_anchors = keras.backend.shape(anchors)[0]

    k = keras.backend.shape(shifts)[0]  # number of base points = feat_h * feat_w

    shifted_anchors = keras.backend.reshape(anchors, [1, number_of_anchors, 4]) + keras.backend.cast(keras.backend.reshape(shifts, [k, 1, 4]), keras.backend.floatx())
    shifted_anchors = keras.backend.reshape(shifted_anchors, [k * number_of_anchors, 4])

    return shifted_anchors

def _mkanchors(ws, hs, x_ctr, y_ctr):
    """
    Given a vector of widths (ws) and heights (hs) around a center
    (x_ctr, y_ctr), output a set of anchors (windows).
    """

    col1 = keras.backend.reshape(x_ctr - 0.5 * (ws - 1), (-1, 1))
    col2 = keras.backend.reshape(y_ctr - 0.5 * (hs - 1), (-1, 1))
    col3 = keras.backend.reshape(x_ctr + 0.5 * (ws - 1), (-1, 1))
    col4 = keras.backend.reshape(y_ctr + 0.5 * (hs - 1), (-1, 1))
    anchors = keras.backend.concatenate((col1, col2, col3, col4), axis=1)

    return anchors

def _ratio_enum(anchor, ratios):
    """
    Enumerate a set of anchors for each aspect ratio wrt an anchor.
    """
    w, h, x_ctr, y_ctr = _whctrs(anchor)
    size = w * h
    size_ratios = size / ratios
    ws = keras.backend.round(keras.backend.sqrt(size_ratios))
    hs = keras.backend.round(ws * ratios)
    anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
    return anchors

def _scale_enum(anchor, scales):
    """
    Enumerate a set of anchors for each scale wrt an anchor.
    """

    w, h, x_ctr, y_ctr = _whctrs(anchor)
    ws = keras.backend.expand_dims(w, 1) * scales
    hs = keras.backend.expand_dims(h, 1) * scales
    anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
    return anchors

def _whctrs(anchor):
    """
    Return width, height, x center, and y center for an anchor (window).
    """
    w = anchor[:, 2] - anchor[:, 0] + 1
    h = anchor[:, 3] - anchor[:, 1] + 1
    x_ctr = anchor[:, 0] + 0.5 * (w - 1)
    y_ctr = anchor[:, 1] + 0.5 * (h - 1)
    return w, h, x_ctr, y_ctr
