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
# -*- coding: utf-8 -*-
# @Time    : 8/9/2018 4:34 PM
# @Author  : sunyonghai
# @Software: ZJ_AI
# -----------------------------------------------------

import numpy as np
from math import *
import cv2
from skimage import exposure


def _horizontal_flip_im(im):
    return im[:, ::-1, :]

def _horizontal_flip_boxes(annotations, image_shape):
    oldx1 = annotations[:,0].copy()
    oldx2 = annotations[:,2].copy()
    annotations[:, 0] = image_shape[1] - oldx2
    annotations[:, 2] = image_shape[1] - oldx1
    assert (annotations[:, 2] >= annotations[:, 0]).all()
    return annotations

def horizontal_transfor(image,annotations):
    image_shape = image.shape
    image = _horizontal_flip_im(image)
    annotations = _horizontal_flip_boxes(annotations, image_shape)
    return image, annotations

def _vertical_flip_im(im):
    return im[::-1, :, :]

def _vertical_flip_boxes(annotations, image_shape):
    oldx1 = annotations[:,1].copy()
    oldx2 = annotations[:,3].copy()
    annotations[:, 1] = image_shape[0] - oldx2
    annotations[:, 3] = image_shape[0] - oldx1
    assert (annotations[:, 3] >= annotations[:, 1]).all()
    return annotations

def vertical_transfor(image,annotations):
    image = _vertical_flip_im(image)
    image_shape = image.shape
    annotations = _vertical_flip_boxes(annotations, image_shape)
    return image, annotations

def _zoom_boxes(boxes, scale):
    new_boxes = boxes.copy()
    for i in range(len(boxes)):
        xmin, ymin, xmax, ymax = boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3]
        min_x = int(int(xmin) * scale[0])
        min_y = int(int(ymin) * scale[1])
        max_x = int(int(xmax) * scale[0])
        max_y = int(int(ymax) * scale[1])
        new_boxes[i][:4] = [min_x, min_y, max_x, max_y]
    return new_boxes

def _shift_boxes(boxes,size,offset):
    def fix_new_key(key, offset, bound):
        if offset >= 0:
            key = min(key, bound)
        else:
            key = max(0, key)
        return key
    new_boxes = boxes.copy()
    box_count=0
    for i in range(len(boxes)):
        xmin, ymin, xmax, ymax=boxes[i][0],boxes[i][1],boxes[i][2],boxes[i][3]
        xmin = fix_new_key(int(int(xmin) + offset[0]), offset[0], size[1])
        ymin = fix_new_key(int(int(ymin) + offset[1]), offset[1], size[0])
        xmax = fix_new_key(int(int(xmax) + offset[0]), offset[0], size[1])
        ymax = fix_new_key(int(int(ymax) + offset[1]), offset[1], size[0])
        if xmax - xmin != 0 and ymax - ymin != 0:
            new_boxes[box_count][:4]=[xmin,ymin,xmax,ymax]
            box_count+=1
    return new_boxes[:box_count]

def _rotate_boxes(boxes,size,angle):
    def rotate_point(width, height, angle, x, y):
        x1 = (x - (width / 2)) * cos(radians(angle)) + (y - (height / 2)) * sin(radians(angle))
        y1 = (y - height / 2) * cos(radians(angle)) - (x - width / 2) * sin(radians(angle))
        return int(x1), int(y1)
    new_boxes=boxes.copy()
    width, height = size[1], size[0]
    heightNew = int(width * fabs(sin(radians(angle))) + height * fabs(cos(radians(angle))))
    widthNew = int(height * fabs(sin(radians(angle))) + width * fabs(cos(radians(angle))))
    for i in range(boxes.shape[0]):
        x_list = [int(boxes[i][0]),int(boxes[i][2])]
        y_list = [int(boxes[i][1]),int(boxes[i][3])]
        max_x, max_y = 0, 0
        min_x, min_y = widthNew, heightNew
        for x in x_list:
            for y in y_list:
                x1, y1 = rotate_point(width, height, angle, x, y)
                x1 = int(x1 + (widthNew / 2))
                y1 = int(y1 + (heightNew / 2))
                max_x = max([max_x, x1])
                min_x = min([min_x, x1])
                max_y = max([max_y, y1])
                min_y = min([min_y, y1])
        new_boxes[i][:4] = [int(min_x),int(min_y),int(max_x),int(max_y)]
    return new_boxes

def cal_resize_scale(image,scale):
    height=image.size[1]
    scale= height/scale
    return scale

def brightness(image, annotations, gamma):
    image = _bright_adjuest_image(image, gamma)
    return image, annotations

def _bright_adjuest_image(im,gamma):
    im = exposure.adjust_gamma(im, gamma)
    return im

def rotate(image, annotations):
    angle = np.random.choice([0,90,180,270],1)[0]
    image_shape = image.shape
    image = _rotate_image(image, angle)
    annotations = _rotate_boxes(annotations, image_shape, angle)
    return image, annotations

def _rotate_image(img,angle):
    height,width=img.shape[0],img.shape[1]
    heightNew = int(width * fabs(sin(radians(angle))) + height * fabs(cos(radians(angle))))
    widthNew = int(height * fabs(sin(radians(angle))) + width * fabs(cos(radians(angle))))

    matRotation = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1)

    matRotation[0, 2] += (widthNew - width) / 2  # 重点在这步，目前不懂为什么加这步
    matRotation[1, 2] += (heightNew - height) / 2  # 重点在这步

    imgRotation = cv2.warpAffine(img, matRotation, (widthNew, heightNew), borderValue=(0, 0, 0))
    return imgRotation

def shift(image, annotations,offset_range):
    image_shape = image.shape
    offset = [int(image_shape[1]* float(offset_range)), int(image_shape[0]* float(offset_range))]
    image = _shift_image(image, offset=offset)
    annotations = _shift_boxes(annotations,image_shape,offset)
    return image, annotations

def _shift_image(image, offset, isseg=False):
    from scipy.ndimage.interpolation import shift
    order = 0
    return shift(image, (int(offset[1]), int(offset[0]), 0), order=order, mode='nearest')

def _zoom_image(image, factor_x,factor_y, isseg=False):
    from scipy.ndimage import interpolation
    order = 0 if isseg == True else 3
    newimg = interpolation.zoom(image, (float(factor_y), float(factor_x), 1.0), order=order, mode='nearest')
    return newimg

def get_crop_bbox(img_size,crop_size,position_index):
    crop_bbox=np.zeros((5,4),dtype=int)
    crop_bbox[0]=np.array((0,0,crop_size[0],crop_size[1]))
    crop_bbox[1]=np.array((0,img_size[1]-crop_size[1],crop_size[0],img_size[1]))
    crop_bbox[2]=np.array((img_size[0]-crop_size[0],0,img_size[0],crop_size[1]))
    crop_bbox[3]=np.array((img_size[0]-crop_size[0],img_size[1]-crop_size[1],img_size[0],img_size[1]))
    crop_bbox[4]=np.array((int(img_size[0]/2-crop_size[0]/2),int(img_size[1]/2-crop_size[1]/2),
                           int(img_size[0]/2+crop_size[0]/2),int(img_size[1]/2+crop_size[1]/2)))
    return crop_bbox[position_index]

def random_crop(image, annotations,crop_range):
    image_shape = image.shape
    position_index = np.random.choice(list(range(5)), 1)[0]
    crop_size = [int(image_shape[1]*crop_range),int(image_shape[0]*crop_range)]
    image,crop_box = random_crop_image(image,crop_size,position_index)
    annotations = cal_random_crop_box(annotations, crop_box)
    return image,annotations


def create_crop_bbox(img_size,crop_size,position_index):
    crop_bbox=np.zeros((5,4),dtype=int)
    crop_bbox[0]=np.array((0,0,crop_size[0],crop_size[1]))
    crop_bbox[1]=np.array((0,img_size[1]-crop_size[1],crop_size[0],img_size[1]))
    crop_bbox[2]=np.array((img_size[0]-crop_size[0],0,img_size[0],crop_size[1]))
    crop_bbox[3]=np.array((img_size[0]-crop_size[0],img_size[1]-crop_size[1],img_size[0],img_size[1]))
    crop_bbox[4]=np.array((int(img_size[0]/2-crop_size[0]/2),int(img_size[1]/2-crop_size[1]/2),
                           int(img_size[0]/2+crop_size[0]/2),int(img_size[1]/2+crop_size[1]/2)))
    return crop_bbox[position_index]

def _resize_box(boxes,scale):
    for i in range(len(boxes)):
        x1,y1,x2,y2=boxes[i][0],boxes[i][1],boxes[i][2],boxes[i][3]
        x1=int(x1/scale)
        y1 = int(y1 / scale)
        x2 = int(x2 / scale)
        y2 = int(y2 / scale)
        boxes[i]=[x1,y1,x2,y2]
    return boxes


def crop(img,box):
    img=img[box[1]:box[3], box[0]:box[2]]
    return img

def cal_scale(size,scale):
    scale= min(size[0],size[1])/scale
    return scale

def random_crop_image(img, crop_size,position_index):
    img_size=(img.shape[1],img.shape[0])
    crop_box = get_crop_bbox(img_size, crop_size,position_index)
    crop_img = crop(img, crop_box)
    return crop_img,crop_box


def cal_random_crop_box(boxes,crop_box):
    new_boxes = boxes.copy()
    count=0
    for i in range(len(boxes)):
        xmin,ymin,xmax,ymax = boxes[i][0],boxes[i][1],boxes[i][2],boxes[i][3]
        x1,y1,x2,y2=crop_box[0],crop_box[1],crop_box[2],crop_box[3]
        min_x =min(max(x1,int(xmin)),x2)-x1
        min_y = min(max(y1, int(ymin)), y2)-y1
        max_x = min(max(x1, int(xmax)), x2)-x1
        max_y = min(max(y1, int(ymax)), y2)-y1
        if max_x-min_x>0 and max_y-min_y>0:
            new_boxes[count][:4]=[min_x,min_y,max_x,max_y]
            count += 1
    return new_boxes[:count]

def resize_box(boxes,scala):
    for i in range(len(boxes)):
        x1,y1,x2,y2=boxes[i][0],boxes[i][1],boxes[i][2],boxes[i][3]
        boxes[i]=[int(int(x1)/scala),int(int(y1)/scala),int(int(x2)/scala),int(int(y2)/scala)]
    return boxes