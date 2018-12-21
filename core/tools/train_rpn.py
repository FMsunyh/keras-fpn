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
# @Time    : 11/9/2018 3:54 PM
# @Author  : Firmin.Sun (fmsunyh@gmail.com)
# @Software: ZJ_AI
# -----------------------------------------------------
# -*- coding: utf-8 -*-
import argparse
import os

import keras
import keras.preprocessing.image

from core.models import VGG16RPN
from core.preprocessing import PascalVocGenerator
from core.utils.config import load_setting_cfg

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '1'

def create_model():
    # image = keras.layers.Input((1000, 600, 3))
    image = keras.layers.Input((None, None, 3))
    gt_boxes = keras.layers.Input((None, 5))
    return VGG16RPN([image, gt_boxes], num_classes=2, weights=None)

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Tensorflow Faster R-CNN demo')

    parser.add_argument('--pascal_path', help='Path to dataset directory (ie. /tmp/VOCdevkit).',required=False,
                               default=['VOC2007'],type=list)
    parser.add_argument('--root_path', help='Size of the batches.', default=os.path.join(os.path.dirname(__file__), '../../'), type=str)
    parser.add_argument('--gpu', help='Id of the GPU to use (as reported by nvidia-smi).', default=0, type=int)
    parser.add_argument('--epochs', help='num of the epochs.', default=100, type=int)
    parser.add_argument('--tag', help='filename of the output.', default='voc', type=str)
    parser.add_argument('--classes_path', help='Path to classes directory (ie. /tmp/com_classes_463.txt).',
                               default='voc_classes.txt',type=str)
    parser.add_argument('--weight_path', help='Path to classes directory (ie. /tmp/com_classes_463.h5).',
                        default='', type=str)
    parser.add_argument('--batch_size', help='Size of the batches.', default=1, type=int)

    parser.add_argument('--tensorboard-dir', help='Log directory for Tensorboard output', default='./logs')
    parser.add_argument('--no-snapshots',    help='Disable saving snapshots.', dest='snapshots', action='store_false')
    parser.add_argument('--snapshot-path',   help='Path to store snapshots of models during training (defaults to \'./snapshots\')',default=os.path.join(os.path.dirname(__file__), '../../snapshots'))
    parser.add_argument('--evaluation',      help='',default=os.path.join(os.path.dirname(__file__), '../../snapshots'))

    # args = parser.parse_args()
    args = check_args(parser.parse_args())
    return args


def check_args(parsed_args):
    #TODO check the args
    # reload parse arguments
    args = load_setting_cfg(parsed_args)

    return args

if __name__ == '__main__':
    # parse arguments
    args = parse_args()

    # create the model
    print('Creating model, this may take a second...')
    model = create_model()

    # compile model (note: set loss to None since loss is added inside layer)
    model.compile(loss=None, optimizer=keras.optimizers.adam(lr=1e-5, clipnorm=0.001))

    # print model summary
    print(model.summary(line_length=250))

    # create image data generator objects
    train_image_data_generator = dict(
        rescale=1.0 / 255.0,
        horizontal_flip=True,
        vertical_flip=True,
        rotate=True,
        random_crop=False,
        brightness=1,
        shift_range=0.1,
        zoom_range=0,
    )
    test_image_data_generator = dict(
        rescale=1.0 / 255.0,
    )

    # create a generator for training data
    train_generator = PascalVocGenerator(
        args,
        'trainval',
        transform_generator = train_image_data_generator
    )

    # create a generator for testing data
    test_generator = PascalVocGenerator(
        args,
        'test',
        transform_generator = test_image_data_generator
    )

    # start training
    batch_size = 1
    model.fit_generator(
        generator=train_generator,
        steps_per_epoch=len(train_generator.image_names) // batch_size,
        epochs=100,
        verbose=1,
        validation_data=test_generator,
        validation_steps=len(test_generator.image_names) // batch_size,
        callbacks=[
            keras.callbacks.ModelCheckpoint(os.path.join(args.root_path, 'snapshots/vgg16_rpn_voc_best.h5'), monitor='val_loss', verbose=1, mode='min', save_best_only=True),
            keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, verbose=1, mode='auto', epsilon=0.0001, cooldown=0, min_lr=0),
        ],
    )

    # store final result too
    model.save('snapshots/vgg16_voc_final.h5')


    '''
    cd tools
    python train_rpn.py pascal /home/syh/train_data/VOCdevkit/VOC2007
    '''