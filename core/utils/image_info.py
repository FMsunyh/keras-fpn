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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from lxml.etree import Element, SubElement, tostring
from xml.dom.minidom import parseString
import os
import cv2

class ImageInfo(object):
    def __init__(self,width,height,path,name,image_extension,channel=3):
        self.width = width
        self.height = height
        self.path = path
        self.name = name
        self.image_extension = image_extension
        self.channel = channel

    def save_image(self,out_path, image):
        # try:
        #     image = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
        # except Exception as ex:
        #     print(out_path)
        #
        try:
            if out_path is not None:
                dir = os.path.dirname(out_path)
                if not os.path.exists(dir):
                    os.makedirs(dir)
                cv2.imwrite(out_path, image)
        except Exception as ex:
            print(ex)

    def save_annotations(self,save_dir, boxes, labels):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        dom = self.make_xml( boxes, labels)
        xml_path = os.path.join(save_dir, self.name + '.xml')
        with open(xml_path, 'w+') as f:
            dom.writexml(f, addindent='', newl='', encoding='utf-8')

    def make_xml(self, boxes, labels):
        node_root = Element('annotation')
        node_folder = SubElement(node_root, 'folder')
        node_folder.text = 'JPEGImages'
        node_filename = SubElement(node_root, 'filename')
        node_filename.text = self.name + '.' + self.image_extension

        node_path = SubElement(node_root, 'path')
        node_path.text = self.path

        node_size = SubElement(node_root, 'size')
        node_width = SubElement(node_size, 'width')
        node_width.text = str(self.width)

        node_height = SubElement(node_size, 'height')
        node_height.text = str(self.height)

        node_depth = SubElement(node_size, 'depth')
        node_depth.text = str(self.channel)

        node_segmented = SubElement(node_root, 'segmented')
        node_segmented.text = '0'

        for i in range(len(labels)):
            label = labels[i]
            b     = boxes[i].split(',')
            node_object = SubElement(node_root, 'object')
            node_name = SubElement(node_object, 'name')
            caption = "{}".format(label)
            node_name.text = caption

            node_pose = SubElement(node_object, 'pose')
            node_pose.text = 'Unspecified'

            node_truncated = SubElement(node_object, 'truncated')
            node_truncated.text = '0'

            node_difficult = SubElement(node_object, 'difficult')
            node_difficult.text = '0'

            node_bndbox = SubElement(node_object, 'bndbox')
            node_xmin = SubElement(node_bndbox, 'xmin')
            node_xmin.text = str(int(b[0]))

            node_ymin = SubElement(node_bndbox, 'ymin')
            node_ymin.text = str(int(b[1]))

            node_xmax = SubElement(node_bndbox, 'xmax')
            node_xmax.text = str(int(b[2]))

            node_ymax = SubElement(node_bndbox, 'ymax')
            node_ymax.text = str(int(b[3]))

        xml = tostring(node_root, pretty_print=True)
        dom = parseString(xml)

        return dom