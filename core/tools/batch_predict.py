#!/usr/bin/env python3
# -----------------------------------------------------
# -*- coding: utf-8 -*-
# @time : 18-12-5
# @Author  : jaykky
# @Software: ZJ_AI
# -----------------------------------------------------
import os
import json
import argparse

import core.tools.predict
from core.utils.evaluate import Evaluate

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '0'


def load_classes(path):
    classes_dict = {}
    try:
        if os.path.splitext(path)[1]=='.txt':
            classes_list = [line.replace('\n','').replace('\r','').strip()  for line in open(path,'r').readlines()]
            for index,label in enumerate(classes_list):
                classes_dict[label] = index
        elif os.path.splitext(path)[1]=='.json':
            classes_dict = json.load(open(path,'r'))
        else:
            raise NotImplementedError
    except:
        print('can not load classes file ,path of {}'.format(path))
    assert len(classes_dict) > 0, 'Please check your classes file again ,path of {}'.format(path)
    return classes_dict

def get_classes(classes_path='/home/hyl/data/ljk/github-pro/keras_frcnn/com_classes_463.txt'):
    classes=load_classes(classes_path)
    classes = sorted(classes.items(),key=lambda d:d[1])
    classes = [cls[0] for cls in classes]
    return classes

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Tensorflow Faster R-CNN demo')

    parser.add_argument('--package_path', help='Size of the batches.',
                        default= ['fruit_data'], type=list)
    parser.add_argument('--weight_path', help='Size of the batches.',
                        default='/home/hyl/keras_frcnn/snapshots/vgg16_frcnn_fruit_best.h5', type=str)
    parser.add_argument('--classes_path', help='Path to classes directory (ie. /tmp/com_classes_463.txt).',
                               default='/home/hyl/data/ljk/github-pro/keras_frcnn/com_classes_463.txt',type=str)
    parser.add_argument('--batch-size', help='Size of the batches.', default=1, type=int)

    args = parser.parse_args()

    return args

def main():
    args = parse_args()
    classes=get_classes(classes_path=args.classes_path)
    weight_path=args.weight_path
    im_paths =[os.path.join(os.path.abspath('.'),'data/predict_data',pkg,'JPEGImages',path) for pkg in args.package_path for path in os.listdir(os.path.join(os.path.abspath('.'),'data/predict_data',pkg,'JPEGImages')) if os.path.splitext(path)[1]=='.jpg']
    core.tools.predict.Predict(classes, weight_path)(inputs=im_paths)
    Evaluate(classes,weight_path)([os.path.join(os.path.abspath('.'),'data/predict_data',pkg) for pkg in args.package_path])

if __name__ == '__main__':
    main()