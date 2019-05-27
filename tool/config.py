# !/usr/bin/python
# -*- coding:utf-8 -*-
class Config():
    def __init__(self, files, cfg_file=None, weight_file=None, lr=1e-3, weight_decay=0.0005,
                 num_cls=20, img_size=608, read_img_size=416, size_num=8,
                 batch_size_per_GPU=8, gpus=1, crop_iou=0.7, keep_ratio=0.2, jitter_ratio=[0.3], stride=[32, 16, 8],
                 ignore_thresh=0.7
                 ):
        self.files = files
        self.cfg_file = cfg_file
        self.weight_file = weight_file
        self.lr = lr
        self.weight_decay = weight_decay
        self.num_cls = num_cls
        self.img_size = img_size
        self.read_img_size = read_img_size
        self.size_num = size_num
        self.batch_size_per_GPU = batch_size_per_GPU
        self.gpus = gpus

        self.crop_iou = crop_iou
        self.keep_ratio = keep_ratio
        self.jitter_ratio = jitter_ratio
        self.stride = stride
        self.ignore_thresh = ignore_thresh
        print('YOLOv3')
        print('==============================================================')

        print('files:\t', self.files)
        print('cfg_file :\t', self.cfg_file)
        print('weight_file:\t', self.weight_file)
        print('lr:\t', self.lr)
        print('weight_decay:\t', self.weight_decay)
        print('num_cls:\t', self.num_cls)
        print('img_size:\t', self.img_size)
        print('read_img_size:\t', self.read_img_size)
        print('size_num:\t', self.size_num)
        print('batch_size_per_GPU:\t', self.batch_size_per_GPU)
        print('gpus:\t', self.gpus)
        print('==============================================================')
        print('crop_iou:\t', self.crop_iou)
        print('keep_ratio:\t', self.keep_ratio)
        print('jitter_ratio:\t', self.jitter_ratio)
        print('stride:\t', self.stride)
        print('ignore_thresh:\t', self.ignore_thresh)
        print('==============================================================')
