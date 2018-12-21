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
# @Time    : 11/23/2018 6:13 PM
# @Author  : Firmin.Sun (fmsunyh@gmail.com)
# @Software: ZJ_AI
# -----------------------------------------------------
# -*- coding: utf-8 -*-

import keras
import tensorflow as tf
from core.backend import bbox_transform_inv

class Proposal(keras.layers.Layer):
    def __init__(self,training=True, nms_thresh=0.8,
                 n_train_pre_nms=12000,
                 n_train_post_nms=2000,
                 n_test_pre_nms=6000,
                 n_test_post_nms=300,
                 min_size=16, *args, **kwargs):
        self.training = training
        self.nms_thresh = nms_thresh
        self.n_train_pre_nms = n_train_pre_nms
        self.n_train_post_nms = n_train_post_nms
        self.n_test_pre_nms = n_test_pre_nms
        self.n_test_post_nms = n_test_post_nms
        self.min_size = min_size

        super(Proposal, self).__init__(*args, **kwargs)

    def call(self, inputs, **kwargs):
        '''
        # TODO: Fix usage of batch index
        # 1. generate proposals from bbox deltas and shifted anchors
        # 2. clip predicted boxes to image
        # 3. remove predicted boxes with either height or width < threshold
        # 4. sort all (proposal, score) pairs by score from highest to lowest
        # 5. take top pre_nms_topN
        # 6. apply nms (e.g. threshold = 0.7)
        # 7. return the top proposals (-> RoIs top)
        :param inputs:
        classification  shape:(batch_size, S,1), from rpn
        regression      shape:(batch_size, S,4), from rpn
        anchors         shape:(batch_size, R,4), from shift all anchors
        image_shape     shape:(2,),  origin image size
        :param kwargs:
        :return: proposal_boxes shape (1, N, 4), N=2000/300
        '''
        classification, regression, anchors, image_shape = inputs

        # TODO batch size > 1
        classification = classification[0]
        regression = regression[0]
        anchors = anchors[0]

        scale = 1.0
        # 1.generator proposal
        regression = tf.reshape(regression, shape=(-1, 4))

        # transfor into proposal with predict shift and anchors
        proposal_boxes = bbox_transform_inv(anchors, regression)

        # 2.clip predicted boxes to image
        proposal_boxes = proposal_boxes[0]
        proposal_boxes = self.clip_boxes(proposal_boxes, image_shape)

        # 3.remove predicted boxes with either height or width < threshold
        proposal_boxes, indices = self.filter_boxes(proposal_boxes, self.min_size * scale)

        classification = classification[..., (classification.shape[-1] // 2)]
        classification = keras.backend.reshape(classification, (-1, 1))
        classification = keras.backend.gather(classification, indices)

        # 5.take top pre_nms_topN
        if self.training:
            n_pre_nms = self.n_train_pre_nms
            n_post_nms = self.n_train_post_nms
        else:
            n_pre_nms = self.n_test_pre_nms
            n_post_nms = self.n_test_post_nms

        # # 4.sort all (proposal, score) pairs by score from highest to lowest
        # _, indices = tf.nn.top_k(classification[:, 0], k=n_pre_nms)
        #
        # # 6.apply nms (e.g. threshold = 0.7)
        # classification  = tf.gather(classification, indices)
        # proposal_boxes = tf.gather(proposal_boxes, indices)

        classification = keras.backend.flatten(classification)
        proposal_boxes, classification_target = self.nms_choice_proposal(proposal_boxes, classification, n_post_nms)

        # proposal_boxes = tf.Print(proposal_boxes, [tf.shape(proposal_boxes)], ' proposal_boxes', summarize=20)
        out = tf.expand_dims(proposal_boxes, axis=0)

        return [out]

    def compute_output_shape(self, input_shape):
        if self.training:
            return [(1, self.n_train_post_nms , 4)]
        else:
            return [(1, self.n_test_post_nms, 4)]

    def compute_mask(self, inputs, mask=None):
        return [None]

    def get_config(self):
        return {
            'n_train_pre_nms'    : self.n_train_pre_nms,
            'nms_thresh'         : self.nms_thresh,
            'n_train_post_nms'   : self.n_train_post_nms,
            'n_test_pre_nms'     : self.n_test_pre_nms,
            'n_test_post_nms'    : self.n_test_post_nms,
            'min_size'           : self.min_size,
        }

    def filter_boxes(self, proposal_boxes, minimum):
        '''
        Filters proposed RoIs so that all have width and height at least as big as  minimum
        :param proposal_boxes: shape (n, 4)
        :param minimum:
        :return:
        '''

        # minimum = tf.cast(minimum, tf.int32)
        ws = proposal_boxes[:, 2] - proposal_boxes[:, 0] + 1
        hs = proposal_boxes[:, 3] - proposal_boxes[:, 1] + 1

        indices = tf.where((ws >= minimum) & (hs >= minimum))
        indices = keras.backend.flatten(indices)
        proposal_boxes = keras.backend.gather(proposal_boxes, indices)
        return proposal_boxes, indices

    def nms_choice_proposal(self, bboxes, classification, n_post_nms):
        '''
        NMS choice some proposal from all proposal box
        :param bboxes: shape (n, 4)
        :param classification: rpn classification, shape (n, 1)
        :param n_post_nms: train:2000 ,test:300
        :return:
        '''

        indices = tf.image.non_max_suppression(bboxes, classification, max_output_size=n_post_nms,iou_threshold=self.nms_thresh)

        proposal_boxes = tf.gather(bboxes,indices)
        classification = tf.gather(classification,indices)

        return proposal_boxes, classification

    def clip_boxes(self, bboxes, image_shape):
        '''
        clip boxes if box out of image shape
        :param bboxes: shape (n, 4)
        :param image_shape: H,W shape (2, )
        :return:
        '''
        # bboxes = tf.cast(bboxes, tf.int32)
        image_shape = tf.cast(image_shape, tf.float32)
        b0 = tf.maximum(tf.minimum(bboxes[:, 0], image_shape[1]), 0.)
        b1 = tf.maximum(tf.minimum(bboxes[:, 1], image_shape[0]), 0.)
        b2 = tf.maximum(tf.minimum(bboxes[:, 2], image_shape[1]), 0.)
        b3 = tf.maximum(tf.minimum(bboxes[:, 3], image_shape[0]), 0.)
        return tf.stack([b0, b1, b2, b3], axis=1)
