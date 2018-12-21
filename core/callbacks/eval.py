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
# @Time    : 12/10/2018 2:38 PM
# @Author  : Firmin.Sun (fmsunyh@gmail.com)
# @Software: ZJ_AI
# -----------------------------------------------------
# -*- coding: utf-8 -*-
import keras
from core.utils.eval import evaluate


class Evaluate(keras.callbacks.Callback):
    def __init__(self, weight, generator, iou_threshold=0.5, score_threshold=0.05, max_detections=100, save_path=None, tensorboard=None, verbose=1):
        """
        Evaluate a given dataset using a given model at the end of every epoch during training.
        :param weight          : The weight path of model.
        :param generator       : The generator that represents the dataset to evaluate.
        :param iou_threshold   : The threshold used to consider when a detection is positive or negative.
        :param score_threshold : The score confidence threshold to use for detections.
        :param max_detections  : The maximum number of detections to use per image.
        :param save_path       : The path to save images with visualized detections to.
        :param tensorboard     : Instance of keras.callbacks.TensorBoard used to log the mAP value.
        :param verbose         : Set the verbosity level, by default this is set to 1.
        :param
        """
        self.weight = weight
        self.generator = generator
        self.iou_threshold = iou_threshold
        self.score_threshold = score_threshold
        self.max_detections = max_detections
        self.save_path = save_path
        self.tensorboard = tensorboard
        self.verbose = verbose

        super(Evaluate, self).__init__()

    def on_epoch_end(self, epoch, logs={}):

        # load the best weight.
        print('Loading weight, this may take a second...')
        self.model.load_weights(self.weight)
        # run evaluation
        print('Strat to evaluate,this may take a long time...')
        average_precisions = evaluate(
            self.generator,
            self.model,
            iou_threshold=self.iou_threshold,
            score_threshold=self.score_threshold,
            max_detections=self.max_detections,
            save_path=self.save_path
        )

        self.mean_ap = sum(average_precisions.values()) / len(average_precisions)

        if self.tensorboard is not None and self.tensorboard.writer is not None:
            import tensorflow as tf
            summary = tf.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = self.mean_ap
            summary_value.tag = "mAP"
            self.tensorboard.writer.add_summary(summary, epoch)

        if self.verbose == 1:
            for label, average_precision in average_precisions.items():
                print(self.generator.label_to_name(label), '{:.4f}'.format(average_precision))
            print('mAP: {:.4f}'.format(self.mean_ap))


# if __name__ == '__main__':
#     e = Evaluate(weight='/home/syh/keras_frcnn/snapshots/voc/voc_vgg16_final.h5')
#     e.on_epoch_end()
