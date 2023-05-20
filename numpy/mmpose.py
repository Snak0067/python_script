# -*- coding:utf-8 -*-
# @FileName :mmpose.py
# @Time :2023/5/19 23:50
# @Author :Xiaofeng
from mmpose.apis import inference_topdown, init_model
from mmpose.utils import register_all_modules

register_all_modules()

config_file = 'td-hm_hrnet-w48_8xb32-210e_coco-256x192.py'
checkpoint_file = 'hrnet_w48_coco_256x192-b9e0b3ab_20200708.pth'
model = init_model(config_file, checkpoint_file, device='cpu')  # or device='cuda:0'

# 请准备好一张带有人体的图片
results = inference_topdown(model, 'demo.jpg')