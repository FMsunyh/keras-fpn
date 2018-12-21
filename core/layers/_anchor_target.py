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
#!/usr/bin/env python3
# -----------------------------------------------------
# -*- coding: utf-8 -*-
# @time : 18-11-14
# @Author  : jaykky
# @Software: ZJ_AI
# -----------------------------------------------------

import keras
import tensorflow as tf
from core.backend import bbox_transform
from core.utils.iou import compute_overlap

class AnchorTarget(keras.layers.Layer):
    def __init__(self, n_sample=256, pos_ratio=0.5,
                 pos_iou_thresh = 0.7,
                 neg_iou_thresh= 0.3,
                 allowed_border = 0.,
                 rpn_bbox_inside_weight = 1,
                 rpn_bbox_positive_weight= -1.0,
                 clobber_positives = False,
                 *args, **kwargs):
        self.n_sample = n_sample
        self.pos_ratio = pos_ratio
        self.pos_iou_thresh = pos_iou_thresh
        self.neg_iou_thresh = neg_iou_thresh
        self.allowed_border = allowed_border
        self.rpn_bbox_inside_weight = rpn_bbox_inside_weight
        self.rpn_bbox_positive_weight = rpn_bbox_positive_weight
        self.clobber_positives = clobber_positives

        super(AnchorTarget, self).__init__(*args, **kwargs)

    def call(self, inputs, **kwargs):
        '''
        choice batchsize training anchor and calculate those box_target
        :param inputs:
        total_anchors  shape: (N,4) anchors
        image_shape    shape: (2, ) [H,W],origin image size
        gt_boxes       shape: (R,4) target object boxes
        :param kwargs:
        :return: labels: anchors labels, sush as -1,0,1,shape = (N, )
        :return: boxs_reg_target: shift value of training sample anchor,shape = (N, 4)
        '''
        total_anchors, image_shape, gt_boxes = inputs

        # TODO batch size > 1
        anchors = total_anchors[0]
        all_anchors = total_anchors[0]
        gt_boxes = gt_boxes[0]

        # gt_boxes = tf.Print(gt_boxes, [tf.shape(gt_boxes)], '\n gt_boxes', summarize=20)


        # 1.Filter anchor which coordinates overflow input image shape
        indices, anchors = self._inside_image(anchors, image_shape)

        # 2.Calculate iou between anchors and gt_boxes
        overlaps = compute_overlap(anchors, tf.to_float(gt_boxes[:, :4]))

        # 3.choice and tag batch size anchor for training
        labels =  self._create_label(gt_boxes, indices, overlaps)
        labels = self._balance(labels)

        # 4.Recover that shape indentical to all anchors with zero
        labels = self._umap_label(labels, all_anchors, indices)
        bbox_reg_targets = self._bbox_umap_tranform_inv(all_anchors, gt_boxes, indices, overlaps)

        # bbox_reg_targets = tf.Print(bbox_reg_targets, [tf.shape(bbox_reg_targets)], ' anchor_target', summarize=20)
        #
        # indices = tf.where(tf.greater_equal(labels, 1))[:, 0]
        # debug = tf.gather(labels, indices)
        # labels = tf.Print(labels, [debug, tf.shape(debug)], ' labels', summarize=10)

        labels           = keras.backend.expand_dims(labels, axis=0)
        bbox_reg_targets = keras.backend.expand_dims(bbox_reg_targets, axis=0)

        return [labels, bbox_reg_targets]

    def compute_output_shape(self, input_shape):
        return [(1, None), (1, None, 4)]

    def _create_label(self, gt_boxes, indices,  overlaps):
        '''
        label: 1 is positive, 0 is negative, -1 is dont care
        :param indices:
        :param overlaps:
        :return:
        '''

        foreground = tf.ones(tf.shape(indices))
        background = tf.zeros(tf.shape(indices))
        negatives  = foreground * -1
        labels     = negatives

        gt_argmax_overlaps_inds = self._get_gtboxes_max_overlaps(gt_boxes, overlaps)

        max_overlaps = self._get_max_overlaps(overlaps)

        if not self.clobber_positives:
            # assign bg labels first so that positive labels can clobber them
            labels = tf.where(keras.backend.less(max_overlaps, self.neg_iou_thresh), background, labels)

        # fg label: for each gt, anchor with highest overlap
        # gt

        condition = tf.sparse_to_dense(gt_argmax_overlaps_inds, tf.shape(labels, out_type=tf.int64), True, default_value=False)
        labels = tf.where(condition=condition, x=foreground, y=labels)

        # anchor with highest overlap
        labels = tf.where(keras.backend.greater_equal(max_overlaps, self.pos_iou_thresh), x=foreground, y=labels)

        if self.clobber_positives:
            # assign bg labels last so that negative labels can clobber positives
            labels = tf.where(keras.backend.less(max_overlaps, self.neg_iou_thresh), x=background, y=labels)

        return labels

    def _balance(self, labels):
        """
        balance labels by setting some to -1
        :param labels: array of labels (1 is positive, 0 is negative, -1 is dont care)
        :return: array of labels
        """
        # subsample positive labels if too many
        labels = self._subsample_positive_labels(labels)

        # subsample negative labels if too many
        labels = self._subsample_negative_labels(labels)
        return labels

    def _subsample_positive_labels(self, labels):
        """
        Sample half of batchsize positive subsample from anchor with set labels to 1
        :param labels: (N, 1) ndarray of float
        :return: labels: set half of batchsize positive subsample to 1,shape = (N, 1)
        """
        num_fg = tf.to_int32(self.pos_ratio * self.n_sample)
        fg_inds = tf.reshape(tf.where(tf.equal(labels, 1)), shape=[-1])
        fg_inds_size = tf.size(fg_inds)
        labels = tf.cond(fg_inds_size > num_fg, true_fn=lambda: self.sampling_anchors(labels, fg_inds, num_fg), false_fn=lambda: labels)
        return labels

    def _subsample_negative_labels(self, labels):
        '''
        Sample negative subsample from anchor with set labels to 0
        :param labels: (N, 1) ndarray of float
        :return: labels: set negative subsample to 0,shape = (N, 1)
        '''
        num_bg = tf.to_int32(self.n_sample - tf.size(tf.reshape(tf.where(tf.equal(labels, 1)), shape=[-1])))
        bg_inds = tf.reshape(tf.where(tf.equal(labels, 0)), shape=[-1])
        bg_inds_size = tf.size(bg_inds)
        labels = tf.cond(bg_inds_size > num_bg, true_fn=lambda: self.sampling_anchors(labels, bg_inds, num_bg), false_fn=lambda: labels)
        return labels

    def _bbox_umap_tranform_inv(self, all_anchors, gt_boxes, indices, overlaps):
        '''
        Calculate box target and recover that shape indentical to all anchors with zero
        :param all_anchors: (N, 4) ndarray of float
        :param gt_boxes: (R, 4) ndarray of float
        :param indices: (A, 1) ndarray of int,A < N
        :param overlaps: (N, R) ndarray of float
        :return: bbox_reg_targets: (N, 4) ndarray to float,but only (batchsize,4) not equal to 0;
        '''
        max_overlaps = self._get_max_overlaps(overlaps)
        anchors = tf.gather(all_anchors, indices)

        bbox_reg_targets = tf.fill((tf.gather(tf.shape(anchors), [0, 1])), 0.)
        boxes = tf.gather(gt_boxes[:, :4], tf.to_int32(max_overlaps))
        bbox_target = bbox_transform(boxes, anchors)

        indices = tf.reshape(indices, shape=(-1, 1))
        bbox_target = tf.scatter_nd(indices=indices, updates=bbox_target, shape=tf.shape(all_anchors))
        bbox_reg_targets = bbox_target
        return bbox_reg_targets

    def _umap_label(self, labels, all_anchors, indices):
        '''
        Recover that shape indentical to all anchors with zero
        :param labels: (A, ) ndarray of int,A < N
        :param all_anchors: (N ,4) ndarray of float
        :param indices: (A, 1) ndarray of int,A < N
        :return: labels: (N , ) ndarray of int, but only batchsize index not equal to -1
        '''
        _labels = tf.add(labels, 1)

        indices = tf.reshape(indices, shape=(-1, 1))
        _labels = tf.scatter_nd(indices=indices, updates=_labels, shape=[tf.shape(all_anchors)[0]]) # fill 0 except for indices
        labels = tf.subtract(_labels, 1)
        return labels

    def _inside_image(self, boxes, image_shape):
        """
        Filter anchor which coordinates overflow input image shape
        :param boxes: (N, 4) ndarray of float
        :param image_shape: [H,W]
        :return: indices: anchors index which inside image,shape = (A,1) A<=N
        :return: indices: anchors coordinates which inside image,shape = (A,4) A<=N
        """
        image_shape = tf.to_float(image_shape)
        w = image_shape[1]
        h = image_shape[0]

        indices = tf.where(
            (boxes[:, 0] >= -self.allowed_border) &
            (boxes[:, 1] >= -self.allowed_border) &
            (boxes[:, 2] < self.allowed_border + image_shape[1]) &  # width
            (boxes[:, 3] < self.allowed_border + image_shape[0])  # height
        )[:, 0]

        indices = tf.to_int32(indices)
        inside_boxes = tf.gather(boxes, indices)
        return indices, tf.reshape(inside_boxes, [-1, 4])

    def _get_max_overlaps(self, overlaps):
        """
        Get max overlaps which iou between gtboxes and anchors with axis = 1
        :param overlaps: (K, 4) ndarray of float
        :return: max_overlaps: max iou(axis = 1),shape = (K,)
        """
        max_overlaps = tf.reduce_max(overlaps, axis=1)

        # 返回 个数等于anchors
        # anchor 和 n 个 gt boxes 的iou, 取最大值（anchor_id, iou1,iou2.。。） ==》（anchor_id, max_iou）
        # 但是这里出去是1维的
        return max_overlaps

    def _get_gtboxes_max_overlaps(self, gt_boxes, overlaps):
        """
        Get max overlaps index which iou between gtboxes and anchors with axis = 0
        :param overlaps: (K, 4) ndarray of float
        :param gt_boxes: (R, 4) ndarray of float
        :return: gt_argmax_overlaps: index of max iou(axis = 0),shape = (R, 1)
        """

        gt_argmax_overlaps = tf.reduce_max(overlaps, axis=0)

        gt_argmax_overlaps = tf.squeeze(tf.equal(overlaps, gt_argmax_overlaps))
        gt_argmax_overlaps = tf.where(gt_argmax_overlaps)[:, 0]
        gt_argmax_overlaps, _ = tf.unique(gt_argmax_overlaps)
        gt_argmax_overlaps = tf.random_shuffle(gt_argmax_overlaps)[:tf.shape(gt_boxes)[0]]

        gt_argmax_overlaps, _ = tf.nn.top_k(
            gt_argmax_overlaps, k=tf.shape(gt_argmax_overlaps)[-1])
        anchor_indices = tf.reverse(gt_argmax_overlaps, [0])

        # 返回 anchor indices, 和gt boxes 重合度最大的anchor
        # 即个数=gt box的个数
        return anchor_indices

    def sampling_anchors(self,labels, anchor_inds, num_sampling):
        # random sampling positive or negative anchor as rpn batchsize dataset
        disable_inds = tf.random_shuffle(anchor_inds)
        disable_inds = disable_inds[num_sampling : tf.size(anchor_inds)]
        disable_inds, _ = tf.nn.top_k(disable_inds, k=tf.shape(disable_inds)[-1])
        disable_inds = tf.reverse(disable_inds, [0])
        disable_inds = tf.sparse_to_dense(disable_inds, tf.shape(labels, out_type=tf.int64), True, default_value=False)
        _labels = tf.where(condition=tf.reshape(disable_inds, shape=[-1]), x=tf.to_float(tf.fill(tf.shape(labels), -1)), y=labels)
        return _labels