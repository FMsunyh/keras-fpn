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
# @Time    : 11/20/2018 3:22 PM
# @Author  : Firmin.Sun (fmsunyh@gmail.com)
# @Software: ZJ_AI
# -----------------------------------------------------
# -*- coding: utf-8 -*-
import keras
import core.backend
import tensorflow as tf

class RPNLoss(keras.layers.Layer):
    def __init__(self, num_classes=2, sigma=3.0, *args, **kwargs):
        self.num_classes = num_classes
        self.sigma       = sigma

        super(RPNLoss, self).__init__(*args, **kwargs)

    def classification_loss(self,labels,  classification):
        '''
        Calculate rpn classification loss
        :param labels: shape of (batchsize, ) tensor
        :param classification: shape of (batchsize, 2) tensor
        :return: classesification loss with the way of crossentropy
        '''
        cls_loss = keras.backend.sparse_categorical_crossentropy(labels, classification)
        cls_loss = keras.backend.sum(cls_loss)

        ones           = keras.backend.ones_like(labels)
        zeros          = keras.backend.zeros_like(labels)
        assigned_boxes = tf.where(keras.backend.greater_equal(labels, 0), ones, zeros)

        cls_loss = cls_loss / (keras.backend.maximum(1.0, keras.backend.sum(assigned_boxes)))
        # cls_loss = tf.Print(cls_loss,[cls_loss],' rpn_cls_loss',summarize=10)

        return cls_loss

    def regression_loss(self,regression_target, regression):
        '''
        Calculate rpn regression loss
        :param labels: shape of (N ,) tensor
        :param regression_target: true value, shape of (N ,4) tensor
        :param regression: predict value, shape of (N ,4) tensor
        :return: rpn regression loss with the way of smooth_L1_loss
        '''
        regression_diff     = regression - regression_target
        abs_regression_diff = keras.backend.abs(regression_diff)

        """
            ResultLoss = outside_weights * SmoothL1(inside_weights * (bbox_pred - bbox_targets))
            SmoothL1(x) = 0.5 * (sigma * x)^2,    if |x| < 1 / sigma^2
                          |x| - 0.5 / sigma^2,    otherwise
        """
        sigma_2 = self.sigma**2
        smoothL1_sign = tf.to_float(keras.backend.less(abs_regression_diff, 1. / sigma_2))
        reg_loss = keras.backend.pow(abs_regression_diff, 2) * (sigma_2 / 2.) * smoothL1_sign + (abs_regression_diff - (0.5 / sigma_2)) * (1. - smoothL1_sign)
        reg_loss = keras.backend.sum(reg_loss, axis=1)

        # reg_loss = tf.reduce_mean(reg_loss)

        # regression weight set :1 / 256
        reg_loss = keras.backend.sum(reg_loss)
        reg_loss = reg_loss / 256


        # reg_loss = tf.Print(reg_loss,[reg_loss],' rpn_reg_loss',summarize=10)
        return reg_loss

    def call(self, inputs):
        classification, labels, regression, regression_target = inputs
        classification    = keras.backend.reshape(classification, (-1, self.num_classes))
        labels            = keras.backend.reshape(labels, (-1,))
        regression        = keras.backend.reshape(regression, (-1, 4))
        regression_target = keras.backend.reshape(regression_target, (-1, 4))

        # classification choice positive and negative sample
        indices = tf.where(keras.backend.not_equal(labels, -1))
        classification    = tf.gather_nd(classification, indices)
        classification_labels            = tf.gather_nd(labels, indices)

        # regression choice positive sample
        indices = tf.where(keras.backend.equal(labels, 1))
        regression        = tf.gather_nd(regression, indices)
        regression_target = tf.gather_nd(regression_target, indices)

        cls_loss = self.classification_loss(classification_labels, classification)
        self.add_loss(cls_loss)

        reg_loss = self.regression_loss(regression_target, regression)
        self.add_loss(reg_loss)
        return [cls_loss, reg_loss]

    def compute_output_shape(self, input_shape):
        return [(1,), (1,)]

    def compute_mask(self, inputs, mask=None):
        return [None,None]

    def get_config(self):
        return {
            'num_classes' : self.num_classes,
            'sigma'       : self.sigma,
        }

class RCNNLoss(keras.layers.Layer):
    def __init__(self, num_classes=21, sigma=3.0, *args, **kwargs):
        self.num_classes = num_classes
        self.sigma       = sigma

        super(RCNNLoss, self).__init__(*args, **kwargs)

    def classification_loss(self, classification, labels):
        '''
        Calculate RCNN classification loss
        :param classification: label probably,shape of (batchsize, 10)
        :param labels: true label,shape of (batchsize, )
        :return:RCNN classification loss with the way of crossentropy
        '''
        cls_loss = keras.backend.sparse_categorical_crossentropy(labels, classification)
        cls_loss = keras.backend.sum(cls_loss)

        ones = keras.backend.ones_like(labels)
        zeros = keras.backend.zeros_like(labels)
        assigned_boxes = tf.where(keras.backend.greater_equal(labels, 0), ones, zeros)

        cls_loss = cls_loss / (keras.backend.maximum(1.0, tf.to_float(keras.backend.sum(assigned_boxes))))
        # cls_loss = tf.Print(cls_loss,[cls_loss],' rcnn_cls_loss',summarize=10)
        return cls_loss

    def regression_loss(self, regression, regression_target):
        '''
        Calculate RCNN regression loss
        :param labels: shape of (batchsize, ) tensor
        :param regression: true shift value,shape of (batchsize, 4) tensor
        :param regression_target: predict shift value,shape of (batchsize,4) tensor
        :return: RCNN regression loss
        '''

        # regression = tf.Print(regression,[tf.shape(regression)],' regression',summarize=10)
        # regression_target = tf.Print(regression_target,[tf.shape(regression_target)],' regression_target',summarize=10)


        regression_diff     = regression - regression_target
        abs_regression_diff = keras.backend.abs(regression_diff)

        """
            ResultLoss = outside_weights * SmoothL1(inside_weights * (bbox_pred - bbox_targets))
            SmoothL1(x) = 0.5 * (sigma * x)^2,    if |x| < 1 / sigma^2
                          |x| - 0.5 / sigma^2,    otherwise
        """
        sigma_2 = self.sigma**2
        smoothL1_sign = tf.to_float(keras.backend.less(abs_regression_diff, 1. / sigma_2))
        reg_loss = keras.backend.pow(abs_regression_diff, 2) * (sigma_2 / 2.) * smoothL1_sign + (abs_regression_diff - (0.5 / sigma_2)) * (1. - smoothL1_sign)
        reg_loss = keras.backend.sum(reg_loss, axis=1)
        # reg_loss = tf.reduce_mean(reg_loss)


        # regression weight set 1/128
        reg_loss = keras.backend.sum(reg_loss)
        reg_loss = reg_loss / 128
        # reg_loss = tf.Print(reg_loss,[reg_loss],' rcnn_reg_loss',summarize=10)

        return reg_loss

    def call(self, inputs):
        classification, labels, regression, regression_target = inputs

        classification    = keras.backend.reshape(classification, (-1, self.num_classes))
        labels            = keras.backend.reshape(labels, (-1,))
        regression        = keras.backend.reshape(regression, (-1, 4 * self.num_classes))
        regression_target = keras.backend.reshape(regression_target, (-1, 4 * self.num_classes))

        # classification choice positive and negative sample
        indices = tf.where(keras.backend.not_equal(labels, -1))
        classification = tf.gather_nd(classification, indices)
        classification_labels = tf.gather_nd(labels, indices)

        # regression choice positive sample
        indices = tf.where(keras.backend.greater_equal(labels, 1))
        regression = tf.gather_nd(regression, indices)
        regression_target = tf.gather_nd(regression_target, indices)

        cls_loss = self.classification_loss(classification,classification_labels)
        self.add_loss(cls_loss)

        reg_loss = self.regression_loss(regression, regression_target)
        self.add_loss(reg_loss)
        return [cls_loss, reg_loss]

    def compute_output_shape(self, input_shape):
        return [(1,), (1,)]

    def compute_mask(self, inputs, mask=None):
        return [None,None]

    def get_config(self):
        return {
            'num_classes' : self.num_classes,
            'sigma'       : self.sigma,
        }