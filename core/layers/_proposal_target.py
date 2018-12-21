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
#
# import keras
#

import keras
import tensorflow as tf
from core.utils.iou import overlapping
from core.backend import bbox_transform

class ProposalTarget(keras.layers.Layer):
    def __init__(self,num_classes=21, n_sample=128,
                 pos_ratio=0.25, pos_iou_thresh=0.5,
                 neg_iou_thresh_hi=0.3, neg_iou_thresh_lo=0.1,
                 bbox_normalize_target=True, *args, **kwargs):
        self.num_classes = num_classes
        self.n_sample = n_sample
        self.pos_ratio = pos_ratio
        self.pos_iou_thresh = pos_iou_thresh
        self.neg_iou_thresh_hi = neg_iou_thresh_hi
        self.neg_iou_thresh_lo = neg_iou_thresh_lo  # NOTE:default 0.1 in py-faster-rcnn
        self.bbox_normalize_target = bbox_normalize_target


        super(ProposalTarget, self).__init__(*args, **kwargs)
    def call(self, inputs, **kwargs):
        '''
        Assigns ground truth to sampled proposal boxes.
        :param inputs:
        proposal_boxes  shape:(None, S, 4)
        gt_boxes        shape:(None, S, 5)
        :param kwargs:
        :return:
        '''
        proposal_boxes, gt_boxes= inputs

        #TODO batch size > 1

        proposal_boxes = proposal_boxes[0]
        gt_boxes = gt_boxes[0]

        gt_label = tf.to_int32(gt_boxes[:, 4])
        proposal_boxes = tf.concat([proposal_boxes, gt_boxes[:,:4]], axis=0)

        # Get anchors which most close to gtboxes, and gt_boxes which most close to anchors
        argmax_overlaps_inds, max_overlaps, _ = overlapping(proposal_boxes, gt_boxes[:,:4])

        # Sample and tag batchsize proposal for training
        pos_index, neg_index = self._sample(max_overlaps)
        labels, seleted_proposal_boxes = self._create_target(pos_index, neg_index, proposal_boxes, gt_label, argmax_overlaps_inds)

        # Calculate shift value between proposal boxes and gt_boxes
        proposal_boxes_target   = self._bbox_tranfrom(pos_index, neg_index, argmax_overlaps_inds, seleted_proposal_boxes, gt_boxes[:,:4])
        proposal_boxes_target   = self._umap_bbox_tranfrom(labels, proposal_boxes_target)

        # proposal_boxes_target = tf.Print(proposal_boxes_target, [tf.shape(proposal_boxes_target)], ' proposal_boxes_target', summarize=20)

        labels                  = tf.expand_dims(labels, axis=0)
        seleted_proposal_boxes  = tf.expand_dims(seleted_proposal_boxes, axis=0)
        proposal_boxes_target   = tf.expand_dims(proposal_boxes_target, axis=0)

        return  [seleted_proposal_boxes, labels, proposal_boxes_target]

    def compute_output_shape(self, input_shape):
        return [(1, self.n_sample, 4), (1, self.n_sample, 1), (1, self.n_sample, 4 * self.num_classes)]

    def compute_mask(self, inputs, mask=None):
        return [None, None, None]

    def get_config(self):
        return {
            'num_classes'       : self.num_classes,
            'n_sample'          : self.n_sample,
            'pos_ratio'         : self.pos_ratio,
            'pos_iou_thresh'    : self.pos_iou_thresh,
            'neg_iou_thresh_hi' : self.neg_iou_thresh_hi,
            'neg_iou_thresh_lo' : self.neg_iou_thresh_lo,
        }

    def _sample(self, max_overlaps):
        '''
        Sample and tag batchsize proposal for training
        :param max_overlaps: gt_boxes index which most close to anchors,shape of (N, ) tensor
        :return: batchsize proposal index for training
        '''
        pos_index, num_pos_proposal_boxes = self._find_foreground(max_overlaps)
        neg_index = self._find_background(max_overlaps, num_pos_proposal_boxes)

        num_supplement = tf.add(tf.size(pos_index), tf.size(neg_index))

        # if background is not enough, choice more proposal which iou< neg_iou_thresh_lo
        neg_index = tf.cond(tf.less(num_supplement, self.n_sample),
                            true_fn=lambda: self._find_background_supplement(neg_index, max_overlaps, tf.to_int32(self.n_sample - num_supplement)),
                            false_fn=lambda: (neg_index))

        # if all proposal less than batchsize, need to cycle choice from background proposal
        num_supplement = tf.add(tf.size(pos_index),tf.size(neg_index))
        neg_index = tf.cond(tf.less(num_supplement, self.n_sample),
                            true_fn=lambda: self._cycle_choice_proposal(neg_index, tf.to_int32(self.n_sample - num_supplement)),
                            false_fn=lambda: (neg_index))

        return pos_index, neg_index

    def _find_foreground(self, max_overlaps):
        '''
        Find foreground proposal boxes
        :param max_overlaps: gt_boxes index which most close to anchors,shape of (N, ) tensor
        :return: foreground proposal boxes index
        '''
        num_pos_proposal_boxes = tf.to_int32(self.n_sample * self.pos_ratio)
        indices = tf.where(max_overlaps > self.pos_iou_thresh)
        max_num = tf.minimum(tf.size(indices), num_pos_proposal_boxes)
        # indices = tf.cond(tf.greater(max_num, 0), true_fn=lambda: self._random_choice(indices, max_num), false_fn=lambda: self._random_choice(indices, max_num))
        indices = self._random_choice(indices, max_num) # if too large,random choice foreground proposal

        return indices, max_num

    def _find_background(self,max_overlaps, pos_max_num):
        '''
        Find background proposal boxes
        :param max_overlaps: gt_boxes index which most close to anchors,shape of (N, ) tensor
        :param pos_max_num: number of need background proposal
        :return: background proposal boxes index
        '''
        num_peg_proposal_boxes = tf.to_int32(self.n_sample - pos_max_num)
        indices = tf.where(tf.logical_and(tf.less(max_overlaps, self.neg_iou_thresh_hi), tf.greater(max_overlaps, self.neg_iou_thresh_lo)))
        max_num = tf.minimum(tf.size(indices), num_peg_proposal_boxes)
        # indices = tf.cond(tf.greater(max_num, 0), true_fn=lambda: self._random_choice(indices, max_num), false_fn=lambda: self._random_choice(indices, max_num))
        indices = self._random_choice(indices, max_num) # if too more,random choice background proposal

        return indices
    
    def _find_background_supplement(self, neg_index, max_overlaps, num_supplement):
        '''
        if background is not enough, choice more proposal which iou< neg_iou_thresh_lo
        :param neg_index: background proposal index
        :param max_overlaps: gt_boxes index which most close to anchors,shape of (N, ) tensor
        :param num_supplement: sting need to choice number
        :return: background proposal
        '''
        indices = tf.where(tf.less(max_overlaps, self.neg_iou_thresh_lo))
        indices = self._random_choice(indices, num_supplement)

        indices = tf.concat([neg_index,indices], axis=0)

        return indices

    def _cycle_choice_proposal(self, inds, num_supplement):
        '''
        if all proposal less than batchsize, need to cycle choice from background proposal
        :param inds: background proposal index
        :param num_supplement: sting need to choice number
        :return: proposal index
        '''
        indices = inds
        def cond(target_inds, num_supplement, inds): # judgement
            return tf.less(tf.size(target_inds), num_supplement)
            # return True

        def body(target_inds, num_supplement, inds): # random choice one proposal index every loop
            inds_choice = tf.gather(tf.random_shuffle(inds), [0])
            target_inds=tf.concat([target_inds,inds_choice],axis=0)
            return target_inds,num_supplement,inds

        target_inds = tf.gather(tf.random_shuffle(inds), [0])
        target_inds, num_supplement, inds=tf.while_loop(cond=cond,body=body,loop_vars=[target_inds, num_supplement,inds],shape_invariants=[tf.TensorShape([None,]),num_supplement.get_shape(),inds.get_shape()])

        indices = tf.concat([indices, target_inds], axis=0)
        return indices

    def _create_target(self, pos_index, neg_index, proposal_boxes, gt_label, argmax_overlaps_inds):
        '''
        tag target batchsize proposal and labels for training
        :param pos_index: foreground proposal ,shape of (f ,) tensor
        :param neg_index: background proposal ,shape of (b ,) tensor ,note: f+b=128
        :param proposal_boxes: all proposal boxes from proposal layer, shape of (N, 4)
        :param gt_label:
        :param argmax_overlaps_inds:
        :return: label and proposal which have been tag for training
        '''
        proposal_label = tf.gather(gt_label, argmax_overlaps_inds)
        indices = tf.concat([pos_index, neg_index], axis=0)

        labels = tf.concat([tf.gather(proposal_label, pos_index), tf.zeros(tf.shape(neg_index), dtype=tf.int32)], axis=0)
        seleted_proposal_boxes = tf.gather(proposal_boxes, indices)

        return labels, seleted_proposal_boxes

    def _bbox_tranfrom(self, pos_index, neg_index, argmax_overlaps_inds, boxes, gt_boxes):
        '''
        calculate shift value between proposal and gt_boxes
        :param pos_index:
        :param neg_index:
        :param argmax_overlaps_inds:
        :param boxes:
        :param gt_boxes:
        :return:
        '''
        gt_box_assign_to_anchors =  tf.gather(gt_boxes, argmax_overlaps_inds)
        seleted_gt_box_assign_to_anchors = tf.gather(gt_box_assign_to_anchors, tf.concat([pos_index, neg_index], axis=0))

        bbox_target = bbox_transform(seleted_gt_box_assign_to_anchors, boxes)

        # if self.bbox_normalize_target:
        #     # Optionally normalize targets by a precomputed mean and stdev
        #     stdev = tf.ones(shape=tf.shape(bbox_target),dtype=tf.float32) * tf.constant([0.1,0.1,0.2,0.2])
        #     bbox_target = ((bbox_target - tf.zeros(shape=tf.shape(bbox_target),dtype=tf.float32)) / stdev)

        return bbox_target

    def _umap_bbox_tranfrom(self, labels, bbox_target):
        positive_index = tf.reshape(tf.where(tf.greater_equal(labels, 1)), shape=[-1])
        positive_cls = tf.to_int64(tf.gather(labels, positive_index))

        positive_bbox_target = tf.gather(bbox_target, positive_index)
        indices = tf.reshape(tf.to_int32(positive_index * self.num_classes + positive_cls), shape=[-1, 1])
        _bbox_target = tf.scatter_nd(indices, positive_bbox_target, shape=[tf.size(labels) * self.num_classes, 4])
        _bbox_target = tf.to_float(tf.reshape(_bbox_target, shape=[-1, self.num_classes * 4]))

        return _bbox_target

    def _random_choice(self, indices, size):
        '''
        Random choice size proposal
        :param indices: foreground or background proposal index
        :param size: need to choice number, size <= len(indices)
        :return: choice index
        '''
        return tf.random_shuffle(tf.reshape(indices, (-1,)))[:size]