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
import yaml
from easydict import EasyDict as edict

def load_setting_cfg(args):
    with open(os.path.join(args.root_path, 'experiments/cfgs/args_setting.cfg'), 'r') as f:
        setting_cfg = edict(yaml.load(f))
    args_dict = args.__dict__
    for k,v in setting_cfg['TRAIN'].items():
        if k in args_dict.keys():
            args_dict[k]=v
    return args