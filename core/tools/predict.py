#!/usr/bin/env python3
# -----------------------------------------------------
# -*- coding: utf-8 -*-
# @time : 18-12-5
# @Author  : jaykky
# @Software: ZJ_AI
# -----------------------------------------------------
import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import keras
from core.models import VGG16FasterRCNN
from ..utils.image import (resize_image, preprocess_image,read_image_bgr)
from ..utils.image_info import ImageInfo

class Predict(object):
    def __init__(self, classes, weight_path,
                 CONF_THRESH=0.8, NMS_THRESH=0.3,
                 verbose=False, save=True,*args, **kwargs):
        self.CONF_THRESH = CONF_THRESH
        self.NMS_THRESH  = NMS_THRESH
        self.CLASSES     = classes
        self.weight_path = weight_path
        self.save        = save
        self.verbose     = verbose

        super(Predict, self).__init__(*args, **kwargs)

    def build_network(self):
        image = keras.layers.Input((None, None, 3))
        model = VGG16FasterRCNN(image, training=False, num_classes=len(self.CLASSES))
        model.load_weights(self.weight_path)
        return model

    def __call__(self, inputs, **kwargs):
        # def call(self,inputs,**kwargs):
        image_paths = inputs

        self.model = self.build_network()

        output,im_shapes = self.predict_ims(image_paths)

        self.save_output(image_paths,im_shapes,output,self.save)

        return output

    def save_output(self, image_paths,im_shapes,outputs,save_output=False):
        if save_output:
            for index in range(len(image_paths)):
                save_dir = os.path.abspath(os.path.join(image_paths[index],'..','..','Annotations_test'))
                info=ImageInfo(width=im_shapes[index][1],height=im_shapes[index][0],path=os.path.split(image_paths[index])[0],
                               name=os.path.split(image_paths[index])[1][:-4],image_extension='jpg')
                info.save_annotations(save_dir=save_dir,boxes=outputs[index][1],labels=outputs[index][0])

    def predict_ims(self, image_paths):
        outputs = []
        im_shapes = []
        # for im_path in image_paths:
        for i in tqdm(range(len(image_paths))):
            output,im_shape = self.predict_im(image_paths[i])
            outputs.append(output)
            im_shapes.append(im_shape)
        return outputs,im_shapes

    def predict_im(self, im_path):
        im = read_image_bgr(im_path)
        input, scale = self.computer_input(im)

        im_shape = input.shape[1:3]
        output = self.model.predict(input)

        output = self.computer_output(output, im_shape, scale)
        self.vis_result(im, output,self.verbose)

        return output,im_shape

    def vis_result(self, im, output,verbose):
        if verbose:
            im = im[:, :, ::-1]

            class_name, dets, cls_socre = output
            # print(im)
            fig, ax = plt.subplots(figsize=(12, 12))
            ax.imshow(im, aspect='equal')
            for i in range(len(class_name)):
                bbox = dets[i].split(',')

                ax.add_patch(
                    plt.Rectangle((int(bbox[0]), int(bbox[1])),
                                  int(bbox[2]) - int(bbox[0]),
                                  int(bbox[3]) - int(bbox[1]), fill=False,
                                  edgecolor='red', linewidth=3.5)
                )
                ax.text(int(bbox[0]), int(bbox[1]) - 2, '{}'.format(class_name[i]),
                        bbox=dict(facecolor='blue', alpha=0.5),
                        fontsize=14, color='white')

            ax.set_title(('{} detections with p({} | box)').format(class_name, class_name),fontsize=14)
            plt.axis('off')
            plt.draw()
            plt.show()

    def computer_output(self, output, im_shape, scale):
        scores, pred_shifts, proposal_boxes = output[0][0], output[1][0], output[2][0]
        object_boxes = self.get_object_boxes(proposal_boxes, pred_shifts)

        object_boxes = self.clip_boxes(object_boxes, im_shape)
        class_name, boxes, cls_socre = self.filter_boxes(object_boxes, scores) # filter boxes which score less than thresh
        # boxes, scores = self.nms_filter(object_boxes, scores)
        # print(boxes.shape,scores.shape)

        boxes = self.resize_boxes(boxes, scale)  # input image have been resize,so boxes need to resize to origin image
        return [class_name, boxes, cls_socre]

    def nms_filter(self,object_boxes,object_scores):
        for cls_ind, cls in enumerate(self.CLASSES[1:]):
            cls_ind += 1  # because we skipped background
            cls_boxes = object_boxes[:, 4 * cls_ind:4 * (cls_ind + 1)]
            cls_scores = object_scores[:, cls_ind]
            dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])).astype(np.float32)
            keep = self.nms(dets, self.NMS_THRESH)
            if cls_ind ==1:
                boxes = object_boxes[keep,:]
                scores = object_scores[keep,:]
            else:
                boxes = np.vstack((boxes,object_boxes[keep,:]))
                scores= np.vstack((scores,object_scores[keep,:]))
        return boxes,scores

    def resize_boxes(self, boxes, scale):
        for i in range(len(boxes)):
            xmin, ymin, xmax, ymax = boxes[i].split(',')
            xmin, ymin, xmax, ymax = int(int(xmin) / scale), int(int(ymin) / scale), int(int(xmax) / scale), int(
                int(ymax) / scale),
            boxes[i] = '{},{},{},{}'.format(xmin, ymin, xmax, ymax)
        return boxes

    def get_object_boxes(self, proposal_boxes, pred_shifts):
        object_boxes = self.bbox_transform_inv(proposal_boxes, pred_shifts)
        return object_boxes

    def filter_boxes(self, object_boxes, scores):
        class_name, boxes, cls_socre = [], [], []
        for cls_ind, cls in enumerate(self.CLASSES[1:]):
            cls_ind += 1  # because we skipped background
            cls_boxes = object_boxes[:, 4 * cls_ind:4 * (cls_ind + 1)]
            cls_scores = scores[:, cls_ind]
            dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])).astype(np.float32)
            dets = dets[np.where(dets[:, -1] >= self.CONF_THRESH)[0]]
            keep = self.nms(dets, self.NMS_THRESH)
            dets = dets[keep, :]
            box, classname, flag, score = self._get_thresh_label(cls, dets, thresh=self.CONF_THRESH)
            if flag==-1:
                continue
            for i in range(len(classname)):
                # result_data.append("{},{:.3f},{},{},{},{}".format(classname[i],score[i],int(box[i, 0]),int(box[i, 1]),int(box[i, 2]),int(box[i, 3])))
                class_name.append(classname[i])
                cls_socre.append(score[i])
                boxes.append('{},{},{},{}'.format(int(box[i, 0]), int(box[i, 1]), int(box[i, 2]), int(box[i, 3])))
        return class_name, boxes, cls_socre

    def _get_thresh_label(self, class_name, dets, thresh=0.5):
        """Draw detected bounding boxes."""
        inds = np.where(dets[:, -1] >= thresh)[0]
        boxes = np.zeros((1, 5), dtype=np.float32)
        cls_list = []
        score_list = []
        flag = 1
        if len(inds)==0:
            flag = 0
            return boxes, cls_list, flag, score_list
        count = 0
        for i in inds:
            bbox = list(map(int, dets[i, :4]))
            score = dets[i, -1]
            bbox.append(score)
            bbox = np.array(bbox)
            if count==0:
                boxes[0, :] = bbox
            else:
                boxes = np.row_stack((boxes, bbox))
            count += 1
            cls_list.append(class_name)
            score_list.append(score)
        return boxes, cls_list, flag, score_list

    def computer_input(self, input):
        input = preprocess_image(input)
        input, scale = resize_image(input)
        input = np.expand_dims(input, axis=0)
        return input, scale

    # def bbox_transform_inv(self, boxes, deltas):
    #     if boxes.shape[0]==0:
    #         return np.zeros((0, deltas.shape[1]), dtype=deltas.dtype)
    #
    #     boxes = boxes.astype(deltas.dtype, copy=False)
    #     widths = boxes[:, 2] - boxes[:, 0] + 1.0
    #     heights = boxes[:, 3] - boxes[:, 1] + 1.0
    #     ctr_x = boxes[:, 0] + 0.5 * widths
    #     ctr_y = boxes[:, 1] + 0.5 * heights
    #
    #     dx = deltas[:, 0::4]
    #     dy = deltas[:, 1::4]
    #     dw = deltas[:, 2::4]
    #     dh = deltas[:, 3::4]
    #
    #     pred_ctr_x = dx * widths[:, np.newaxis] + ctr_x[:, np.newaxis]
    #     pred_ctr_y = dy * heights[:, np.newaxis] + ctr_y[:, np.newaxis]
    #     pred_w = np.exp(dw) * widths[:, np.newaxis]
    #     pred_h = np.exp(dh) * heights[:, np.newaxis]
    #
    #     pred_boxes = np.zeros(deltas.shape, dtype=deltas.dtype)
    #     # x1
    #     pred_boxes[:, 0::4] = pred_ctr_x - 0.5 * pred_w
    #     # y1
    #     pred_boxes[:, 1::4] = pred_ctr_y - 0.5 * pred_h
    #     # x2
    #     pred_boxes[:, 2::4] = pred_ctr_x + 0.5 * pred_w
    #     # y2
    #     pred_boxes[:, 3::4] = pred_ctr_y + 0.5 * pred_h
    #
    #     return pred_boxes

    def clip_boxes(self, boxes, im_shape):
        # x1 >= 0
        boxes[:, 0::4] = np.maximum(np.minimum(boxes[:, 0::4], im_shape[1] - 1), 0)
        # y1 >= 0
        boxes[:, 1::4] = np.maximum(np.minimum(boxes[:, 1::4], im_shape[0] - 1), 0)
        # x2 < im_shape[1]
        boxes[:, 2::4] = np.maximum(np.minimum(boxes[:, 2::4], im_shape[1] - 1), 0)
        # y2 < im_shape[0]
        boxes[:, 3::4] = np.maximum(np.minimum(boxes[:, 3::4], im_shape[0] - 1), 0)
        return boxes

    def nms(self, dets, thresh):
        # dets:(m,5)  thresh:scaler
        x1 = dets[:, 0]
        y1 = dets[:, 1]
        x2 = dets[:, 2]
        y2 = dets[:, 3]

        areas = (y2 - y1 + 1) * (x2 - x1 + 1)
        scores = dets[:, 4]
        keep = []

        index = scores.argsort()[::-1]

        while index.size > 0:
            i = index[0]  # every time the first is the biggst, and add it directly
            keep.append(i)

            x11 = np.maximum(x1[i], x1[index[1:]])  # calculate the points of overlap
            y11 = np.maximum(y1[i], y1[index[1:]])
            x22 = np.minimum(x2[i], x2[index[1:]])
            y22 = np.minimum(y2[i], y2[index[1:]])

            w = np.maximum(0, x22 - x11 + 1)  # the weights of overlap
            h = np.maximum(0, y22 - y11 + 1)  # the height of overlap

            overlaps = w * h

            ious = overlaps / (areas[i] + areas[index[1:]] - overlaps)

            idx = np.where(ious <= thresh)[0]

            index = index[idx + 1]  # because index start from 1

        return keep

    def get_config(self):
        return {'CON_THRESH': self.CON_THRESH,
                'NMS_THRESH': self.NMS_THRESH,
                'CLASSES'   : self.CLASSES    }
