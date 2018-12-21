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
# @Time    : 12/17/2018 5:34 PM
# @Author  : Firmin.Sun (fmsunyh@gmail.com)
# @Software: ZJ_AI
# -----------------------------------------------------
# -*- coding: utf-8 -*-
import cv2
import numpy as np

from .colors import label_color

def draw_box(image, box, color, thickness=2):
    raise  NotImplementedError

def draw_boxes(image, boxes, color, thickness=2):
    raise NotImplementedError

def draw_caption(image, box, caption):
    """
    Draws a caption above the box in an image.
    :param image:
    :param box:
    :param caption:
    :return:
    """
    raise  NotImplementedError

def draw_detections(image, boxes, scores, labels, color=None, label_to_name=None, score_threshold=0.5):
    # draw detection boxes
    raise NotImplementedError

def draw_annotations(image, annotations, color=(0, 255, 0), label_to_name=None):
    # draw gt  boxes
    raise NotImplementedError