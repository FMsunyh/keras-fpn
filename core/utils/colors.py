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
# @Time    : 12/14/2018 10:54 AM
# @Author  : Firmin.Sun (fmsunyh@gmail.com)
# @Software: ZJ_AI
# -----------------------------------------------------
# -*- coding: utf-8 -*-
import warnings
import numpy as np


def label_color(label):
    """
    Return a color from a set of predefined colors. Contains N colors in total.
    N: number of labels
    :param label: The label to get the color for.
    :return:
        A list of three values representing a RGB color.
        If no color is defined for a certain label, the color green is returned and a warning is printed.

    """
    if label < len(colors):
        return colors[label]
    else:
        warnings.warn('Label {} has no color, returning default.'.format(label))
        return (0, 255, 0)

def generator_colors(N = 1000):
    '''
    Generator N type of colors
    :param N:
    :return:
    '''
    colors = []
    group_num = round(pow(N, 1/3))

    for x in np.arange(0., 1., 1 / group_num):
        for y in np.arange(0., 1., 1 / group_num):
            for z in np.arange(0., 1., 1 / group_num):
                # color = list((matplotlib.colors.hsv_to_rgb([x, y, z]) * 255).astype(int))
                color = list([int(x * 255), int(y * 255), int(z * 255)])
                # color = list([x, y, z])
                colors.append(color)
    return colors
"""
Generated using:

```
colors = [list((matplotlib.colors.hsv_to_rgb([x, 1.0, 1.0]) * 255).astype(int)) for x in np.arange(0, 1, 1.0 / N)]
shuffle(colors)
pprint(colors)
```
"""
# colors = [
#     [31  , 0   , 255] ,
#     [0   , 159 , 255]
# ]
colors = generator_colors(N = 1000)