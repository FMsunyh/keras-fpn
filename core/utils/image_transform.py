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
# @time : 18-12-10
# @Author  : jaykky
# @Software: ZJ_AI
# -----------------------------------------------------
import random
from core.utils import data_augument

class imageTransfor(object):
    def __init__(self,transfor_dict,random_choice=True):
        '''

        :param transfor_dict:  dict(horizontal_flip = True,
                                    vertical_flip   = True,
                                    rotate          = True,
                                    random_crop     = 0.1,
                                    brightness      = 0.8,
                                    shift_range     = 0.1,
                                    zoom_range      = 0,)
        :param random_choice:
        '''
        self.transfor_dict = transfor_dict
        self.random_choice = random_choice

    def __call__(self, input, **kwargs):
        image, annotations = input

        transfor_way = self.choice_transfor_way()

        image, annotations = self.transfor(image, annotations, transfor_way)

        return image,annotations

    def choice_transfor_way(self):

        transfor_way = self.transfor_dict

        return self.random_choice_one(transfor_way)

    def random_choice_one(self,transfor_way):
        '''
        random choice one transfor way of operator image
        '''
        one_transfor_way = {}
        random_key = list(transfor_way.keys())
        random.shuffle(random_key)
        one_key = random_key[0]
        one_transfor_way[one_key] = transfor_way[one_key]
        return one_transfor_way

    def keep_positive(self,transfor_way):
        positive_transfor_way = {}
        for (key,value) in transfor_way.items():
            if value :
                positive_transfor_way[key] = value

        return positive_transfor_way

    def transfor(self,image, annotations, transfor_way):
        '''
        according to transfor way,transfor image
        :param image:
        :param annotations:
        :param transfor_way:
        :return:
        '''
        for key,value in transfor_way.items():
            if key == 'horizontal_flip' and value:
                image, annotations = data_augument.horizontal_transfor(image, annotations)
            if key == 'vertical_flip' and value:
                image, annotations = data_augument.vertical_transfor(image, annotations)
            if key == 'shift_range' and value<1 and value>0:
                image, annotations = data_augument.shift(image, annotations, value)
            if key == 'rotate' and value:
                image, annotations = data_augument.rotate(image, annotations)
            if key == 'random_crop' and value<1 and value>0:
                data_augument.random_crop(image,annotations,value)
            if key == 'brightness' and value<1 and value >0 :
                print('NotImplementedError')
            if key == 'zoom_range' and value<1 and value >0 :
                print('NotImplementedError')
        return image, annotations


