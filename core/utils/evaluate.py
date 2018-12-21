#!/usr/bin/env python3

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
# @time : 18-12-7
# @Author  : jaykky
# @Software: ZJ_AI
# -----------------------------------------------------
import os
import prettytable as pt
import numpy as np
import xml.etree.ElementTree as ET

class Evaluate(object):
    def __init__(self,classes,model):
        self.CLASSES = classes
        self.ROOT_DIR = os.path.abspath('.')
        self.model = model

    def __call__(self, inputs, **kwargs):
        packages = inputs

        self.eval_result(packages)

    def eval_result(self,packages):
        test_infos=[]
        for package in packages:
            test_xml_path = os.path.join(self.ROOT_DIR, 'data', 'train_data', package, 'Annotations_test')
            true_xml_path = os.path.join(self.ROOT_DIR, 'data', 'train_data', package, 'Annotations')
            test_info = self.eval_whole_acc(test_xml_path, true_xml_path)
            test_info_label = self.eval_label_acc(test_xml_path, true_xml_path, self.CLASSES)
            test_infos.append("{},{},total,{}".format(os.path.split(self.model.split(".")[0])[1], os.path.split(package)[1], test_info))
            for index in range(1,len(self.CLASSES)):
                test_infos.append("{},{},{},{}".format(os.path.split(self.model.split(".")[0])[1], os.path.split(package)[1], self.CLASSES[index],
                                                          test_info_label[index]))
        tb = self.get_tabs(test_infos)
        tb = self.summary_tb(tb, test_infos)
        print(tb[0])
        aug_code = os.path.split(self.model)[1][:8]
        txt_save_path = os.path.abspath(os.path.join(self.model, '..', '{}_batch_test_result'.format(aug_code)))
        if not os.path.exists(txt_save_path):
            os.mkdir(txt_save_path)
        txt_save_path = os.path.join(txt_save_path, self.model.split(".")[0] + "_test_result")
        self.save_tb_in_txt(txt_save_path, tb)

    def get_tabs(self,test_infos):
        tb = pt.PrettyTable()
        tb.field_names = ["model_name", "test_data", 'label', 'presion', 'recall', "detect_num", "actual_num", "tp_num",
                          "fp_num", 'fn_num']
        for test_info in test_infos:
            info = test_info.split(",")
            tb.add_row(info)
        return tb

    def save_tb_in_txt(self,path, tb):
        # tb.field_names = ["模型名称", "测试数据", '精确率', '召回率', "模型识别总数", "实际总数", "正确识别数量", "误识别总数",
        #                   '漏识别总数']
        f = open(path + '.txt', "a+")
        f.write(str(tb))
        f.write('\n')
        f.close()

    def summary_tb(self,tb,test_infos):
        presion,recall,d_num,t_num,tp_num,fp_num,fn_num,count=0,0,0,0,0,0,0,0
        model_name,test_data='','total'
        for test_info in test_infos:
            infos = test_info.split(",")
            if count == 0:
                model_name = infos[0]
            count+=1
            d_num+=int(infos[5])
            t_num+=int(infos[6])
            tp_num+=int(infos[7])
            fp_num+=int(infos[8])
            fn_num+=int(infos[9])
        presion=tp_num/(tp_num+fp_num)
        recall=tp_num/(tp_num+fn_num)
        tb.add_row([model_name,test_data,'total',presion,recall,d_num,t_num,tp_num,fp_num,fn_num])
        return tb

    def get_xml_label_num(self,xmlPath):
        if os.path.exists(xmlPath)!=1:
            print(xmlPath)
        et = ET.parse(xmlPath)
        element = et.getroot()
        element_objs = element.findall('object')
        count = len(element_objs)
        labelList = []
        for element_obj in element_objs:
            node = element_obj.find('name')
            label = node.text
            labelList.append(label)
        return count, labelList

    def eval_whole_acc(self,xmlPath1, xmlPath2):
        xmlFileList1 = []
        xmlFileList2 = []
        for xmlFile in os.listdir(xmlPath1):
            xmlFileList1.append(os.path.join(xmlPath1, xmlFile))
            xmlFileList2.append(os.path.join(xmlPath2, xmlFile))

        tp_sum, fp_sum, fn_sum, d_sum, t_sum = 0, 0, 0, 0, 0
        for i in range(len(xmlFileList1)):
            tp, fp, fn = 0, 0, 0
            xmlFile1 = xmlFileList1[i]
            xmlFile2 = xmlFileList2[i]
            d_labelNum, d_labelList = self.get_xml_label_num(xmlFile1)
            t_labelNum, t_labelList = self.get_xml_label_num(xmlFile2)
            for d_label in d_labelList:
                if d_label in t_labelList:
                    labenIndex = t_labelList.index(d_label)
                    t_labelList.remove(t_labelList[labenIndex])
                    tp += 1
                else:
                    fp += 1
                fn = t_labelNum - tp
            tp_sum += tp
            fp_sum += fp
            fn_sum += fn
            d_sum += d_labelNum
            t_sum += t_labelNum
        prec = tp_sum / (fp_sum + tp_sum)
        recall = tp_sum / (tp_sum + fn_sum)
        return "{},{},{},{},{},{},{}".format(prec, recall, d_sum, t_sum, tp_sum, fp_sum, fn_sum)

    def init_ind(self,class_name):
        tp_sum = np.zeros(class_name, dtype=int)
        fp_sum = np.zeros(class_name, dtype=int)
        fn_sum = np.zeros(class_name, dtype=int)
        d_sum = np.zeros(class_name, dtype=int)
        t_sum = np.zeros(class_name, dtype=int)
        prec = np.zeros(class_name, dtype=float)
        rec = np.zeros(class_name, dtype=float)
        return tp_sum, fp_sum, fn_sum, d_sum, t_sum, prec, rec

    def eval_label_acc(self,xmlPath1, xmlPath2, CLASSES):
        classes = list(CLASSES)
        xmlFileList1 = [os.path.join(xmlPath1, xmlFile) for xmlFile in os.listdir(xmlPath1)]
        xmlFileList2 = [os.path.join(xmlPath2, xmlFile) for xmlFile in os.listdir(xmlPath1)]
        tp_arr, fp_arr, fn_arr, d_sum_arr, t_sum_arr, prec_arr, rec_arr = self.init_ind(len(classes))
        for i in range(len(xmlFileList1)):
            xmlFile1 = xmlFileList1[i]
            xmlFile2 = xmlFileList2[i]
            d_labelList, d_boxes = self.get_xml_label_bnd(xmlFile1)
            t_labelList, t_boxes = self.get_xml_label_bnd(xmlFile2)
            for t_label in t_labelList:
                class_index = classes.index(t_label)
                t_sum_arr[class_index] += 1
            for j, d_label in enumerate(d_labelList):
                class_index = classes.index(d_label)
                d_sum_arr[class_index] += 1
                if d_label in t_labelList:
                    tp_arr[class_index] += 1
                    t_labelList.remove(d_label)

        fp_arr = d_sum_arr - tp_arr
        fn_arr = t_sum_arr - tp_arr
        prec_arr = tp_arr / (tp_arr + fp_arr)
        prec_arr[np.isnan(prec_arr)] = 0
        rec_arr = tp_arr / (tp_arr + fn_arr)
        rec_arr[np.isnan(rec_arr)] = 0
        return ["{},{},{},{},{},{},{}".format(prec_arr[num], rec_arr[num], d_sum_arr[num], t_sum_arr[num],
                                              tp_arr[num], fp_arr[num], fn_arr[num]) for num in range(len(classes))]

    def get_xml_label_bnd(self,xmlPath):
        if os.path.exists(xmlPath)!=1:
            print(xmlPath)
        et = ET.parse(xmlPath)
        element = et.getroot()
        element_objs = element.findall('object')

        labelList = []
        boxes = np.zeros((1, 4), dtype=int)
        for i, element_obj in enumerate(element_objs):
            node = element_obj.find('name')
            label = node.text
            labelList.append(label)
            bbox = element_obj.find('bndbox')
            if i==0:
                boxes[0, :] = np.array(
                    [int(bbox.find('xmin').text), int(bbox.find('ymin').text), int(bbox.find('xmax').text),
                     int(bbox.find('ymax').text)])
            else:
                box = np.array([int(bbox.find('xmin').text), int(bbox.find('ymin').text), int(bbox.find('xmax').text),
                                int(bbox.find('ymax').text)])
                boxes = np.row_stack((boxes, box))
        return labelList, boxes