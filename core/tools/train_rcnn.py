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
import glob

import keras
import keras.preprocessing.image
import tensorflow as tf
from core.callbacks import RedirectModel
from core.callbacks.eval import Evaluate
from core.models import VGG16FasterRCNN
from core.preprocessing import PascalVocGenerator
from core.utils.config import load_setting_cfg
from core.models.vgg16 import VGG16FasterRCNN_bbox

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"


def get_session():
    cfg = tf.ConfigProto()
    cfg.gpu_options.allocator_type = 'BFC'
    # cfg.gpu_options.per_process_gpu_memory_fraction = 0.90
    cfg.gpu_options.allow_growth = True
    return tf.Session(config=cfg)

def set_gpu(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    sess = get_session()
    import keras.backend.tensorflow_backend as ktf
    ktf.set_session(sess)

def create_models(num_classes=21):
    image = keras.layers.Input((None, None, 3))
    gt_boxes = keras.layers.Input((None, 5))

    train_model = VGG16FasterRCNN([image, gt_boxes], num_classes=num_classes)
    prediction_model = VGG16FasterRCNN_bbox(num_classes=num_classes)

    return train_model, prediction_model

def create_callbacks(training_model, prediction_model, validation_generator, args):
    callbacks = []

    # save the model
    if args.snapshots:
        # ensure directory created first; otherwise h5py will error after epoch.
        os.makedirs(args.snapshot_path, exist_ok=True)
        checkpoint_model = keras.callbacks.ModelCheckpoint(
            # os.path.join(
            #     args.snapshot_path,
            #     '{backbone}_{{epoch:02d}}.h5'.format(backbone='vgg16')
            # ),
            os.path.join(args.root_path, 'snapshots', args.tag, args.tag + '_{epoch:02d}.h5'),
            verbose=1,
            monitor='loss',
            save_best_only=True,
            save_weights_only=True,
            mode='min',
        )

        # save the prediction weight
        weight_path = os.path.join(args.root_path, 'snapshots', args.tag, args.tag + '_weight_prediction.h5')
        checkpoint_weight = keras.callbacks.ModelCheckpoint(
            weight_path,
            verbose=1,
            monitor='loss',
            save_best_only=True,
            mode='min',
        )

        checkpoint_model = RedirectModel(checkpoint_model, training_model)
        callbacks.append(checkpoint_model)

        checkpoint_weight = RedirectModel(checkpoint_weight, training_model)
        callbacks.append(checkpoint_weight)

    tensorboard_callback = None

    if args.tensorboard_dir:
        tensorboard_dir = os.path.abspath(os.path.join(args.root_path, args.tensorboard_dir))
        tensorboard_callback = keras.callbacks.TensorBoard(
            log_dir                = tensorboard_dir,
            histogram_freq         = 0,
            batch_size             = args.batch_size,
            write_graph            = True,
            write_grads            = False,
            write_images           = False,
            embeddings_freq        = 0,
            embeddings_layer_names = None,
            embeddings_metadata    = None
        )
        callbacks.append(tensorboard_callback)

    # evaluation
    if args.evaluation and validation_generator:
        evaluation = Evaluate(weight_path, validation_generator, tensorboard=tensorboard_callback)
        evaluation = RedirectModel(evaluation, prediction_model)
        callbacks.append(evaluation)

    callbacks.append(keras.callbacks.ReduceLROnPlateau(
        monitor  = 'loss',
        factor   = 0.1,
        patience = 2,
        verbose  = 1,
        mode     = 'auto',
        min_delta  = 0.0001,
        cooldown = 0,
        min_lr   = 0
    ))

    return callbacks

def create_generators(args):
    # create image data generator objects
    # train_image_data_generator = keras.preprocessing.image.ImageDataGenerator(
    train_image_data_generator = dict(
        rescale=1.0 / 255.0,
        horizontal_flip=True,
        vertical_flip=True,
        rotate=True,
        random_crop=0.1,
        brightness=1,
        shift_range=0.1,
        zoom_range=0,
    )
    # valid_image_data_generator = keras.preprocessing.image.ImageDataGenerator(
    valid_image_data_generator = dict(
        rescale=1.0 / 255.0,
    )

    # create a generator for training data
    train_generator = PascalVocGenerator(
        args,
        'trainval',
        transform_generator=train_image_data_generator
    )

    # create a generator for testing data
    validation_generator = PascalVocGenerator(
        args,
        'test',
        transform_generator=valid_image_data_generator
    )

    return train_generator, validation_generator

def make_dir(root_path,tag):
    snapshot_dir = os.path.join(root_path,'snapshots',tag)
    if not os.path.exists(snapshot_dir):
        os.makedirs(snapshot_dir)
    return snapshot_dir

def load_trained_model(model,args):
    init_epoch = 0
    weight_path = ''

    # set snapshot file if exist
    snapshot_dir = os.path.abspath(os.path.join(args.root_path, 'snapshots', args.tag))
    snapshot_files = [fn for fn in glob.glob(os.path.join(snapshot_dir, args.tag+'_[0-9][0-9].h5'))]
    if len(snapshot_files) > 0:
        snapshot_files.sort(reverse=True)
        weight_path = snapshot_files[0]
        init_epoch  = int(os.path.split(weight_path)[1][-5:-3])

    # reload weight_path if exist although snapshot file is exist
    if args.weight_path and os.path.exists(args.weight_path):
        weight_path = args.weight_path
        init_epoch  = 0

    # load trained model if weight path have been set
    if weight_path != '':
        model.load_weights(weight_path)

    return model, init_epoch

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

def main():
    # parse arguments
    args = parse_args()

    # set gpu
    set_gpu(args)

    train_generator, validation_generator = create_generators(args)

    # create the model
    print('Creating model, this may take a second...')
    train_model, prediction_model = create_models(num_classes=train_generator.num_classes())

    print(train_model.summary(line_length=180))

    # create snapshots dir
    snapshot_dir = make_dir(args.root_path, args.tag)

    # load trained model if exist
    train_model, init_epoch = load_trained_model(train_model, args)

    # compile model (note: set loss to None since loss is added inside layer)
    train_model.compile(loss=None, optimizer=keras.optimizers.adam(lr=1e-5, clipnorm=0.001))

    # create the callbacks
    callbacks = create_callbacks(
        train_model,
        prediction_model,
        validation_generator,
        args,
    )

    # start training
    train_model.fit_generator(
        generator=train_generator,
        steps_per_epoch=len(train_generator.image_names) // args.batch_size,
        # steps_per_epoch=10,
        epochs=int(args.epochs),
        verbose=1,
        initial_epoch=init_epoch,
        callbacks=callbacks
    )

    # store final result too
    train_model.save(os.path.join(args.root_path, 'snapshots', args.tag, args.tag + '_vgg16_final.h5'))

if __name__ == '__main__':
    main()

    '''
    cd tools
    # python train_rcnn.py pascal /home/syh/train_data/VOCdevkit/VOC2007
    python train_rcnn.py
    '''