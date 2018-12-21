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
# @Time    : 12/17/2018 11:02 AM
# @Author  : Firmin.Sun (fmsunyh@gmail.com)
# @Software: ZJ_AI
# -----------------------------------------------------
# -*- coding: utf-8 -*-
import argparse
import os
import sys

import keras
import tensorflow as tf
from models.vgg16 import VGG16FasterRCNN_bbox
from core.preprocessing import PascalVocGenerator
from utils.config import load_setting_cfg
from utils.eval import evaluate

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

def get_session():
    cfg = tf.ConfigProto()
    cfg.gpu_options.allocator_type = 'BFC'
    # cfg.gpu_options.per_process_gpu_memory_fraction = 0.90
    cfg.gpu_options.allow_growth = True
    return tf.Session(config=cfg)

def set_gpu():
    sess = get_session()

    import keras.backend.tensorflow_backend as ktf
    ktf.set_session(sess)

def create_generator(args):

    # test_image_data_generator = keras.preprocessing.image.ImageDataGenerator(
    test_image_data_generator = dict(
        rescale=1.0 / 255.0,
    )

    # create a generator for testing data
    validation_generator = PascalVocGenerator(
        args,
        'test1',
        transform_generator=test_image_data_generator
    )

    return validation_generator


def check_args(parsed_args):
    #TODO check the args

    return parsed_args

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Tensorflow Faster R-CNN demo')

    parser.add_argument('--pascal_path', help='Path to dataset directory (ie. /tmp/VOCdevkit).',required=False,
                               default=['VOC2007'],type=list)
    parser.add_argument('--root_path', help='Size of the batches.', default=os.path.join(os.path.dirname(__file__), '../../'), type=str)
    parser.add_argument('--epochs', help='num of the epochs.', default=100, type=int)
    parser.add_argument('--tag', help='filename of the output.', default='voc', type=str)
    parser.add_argument('--classes_path', help='Path to classes directory (ie. /tmp/com_classes_463.txt).',
                               default='voc_classes.txt',type=str)
    parser.add_argument('--weight_path', help='Path to classes directory (ie. /tmp/com_classes_463.h5).',
                        default='', type=str)
    parser.add_argument('--gpu', help='Id of the GPU to use (as reported by nvidia-smi).',default=0, type=int)

    parser.add_argument('--score-threshold', help='Threshold on score to filter detections with (defaults to 0.05).',
                        default=0.05, type=float)
    parser.add_argument('--iou-threshold', help='IoU Threshold to count for a positive detection (defaults to 0.5).',
                        default=0.5, type=float)
    parser.add_argument('--max-detections', help='Max Detections per image (defaults to 100).', default=100, type=int)
    parser.add_argument('--save-path', help='Path for saving images with detections.')
    parser.add_argument('--image-min-side', help='Rescale the image so the smallest side is min_side.', type=int,
                        default=600)
    parser.add_argument('--image-max-side', help='Rescale the image if the largest side is larger than max_side.',
                        type=int, default=1000)

    args = check_args(parser.parse_args())
    return args


def main(args=None):
    # parse arguments

    args = parse_args()
    args = load_setting_cfg(args)

    # optionally choose specific GPU
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
        set_gpu()

    # make save path if it doesn't exist
    if args.save_path is not None and not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    # create the generator
    generator = create_generator(args)

    # load the model
    print('Loading model, this may take a second...')
    model = VGG16FasterRCNN_bbox(num_classes=generator.num_classes())
    model.load_weights(filepath=args.weight_path)

    # print model summary
    print(model.summary())

    # start evaluation
    print('start evaluate')
    average_precisions = evaluate(
        generator,
        model,
        iou_threshold=args.iou_threshold,
        score_threshold=args.score_threshold,
        max_detections=args.max_detections,
        save_path=args.save_path
    )

    # print evaluation
    for label, average_precision in average_precisions.items():
        print(generator.label_to_name(label), '{:.4f}'.format(average_precision))
    print('mAP: {:.4f}'.format(sum(average_precisions.values()) / len(average_precisions)))

if __name__ == '__main__':
    main()
