from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys
import time
import torch
import numpy as np


class opts(object):
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        # basic experiment setting
        self.parser.add_argument('--exp_id', default='default',
                                 help='dir to save weights and logs')
        self.parser.add_argument('--image', default='',
                                 help='path to image for detecting')
        self.parser.add_argument('--load_model', default='',
                                 help='path to pretrained model')

        # system
        self.parser.add_argument('--gpus', default='0',
                                 help='use comma for multiple gpus')
        self.parser.add_argument('--vis_thresh', type=float, default=0.4,
                                 help='visualization threshold.')

        # model
        self.parser.add_argument('--arch', default='resnet18',
                                 help='model architecture. ')

        # input
        self.parser.add_argument('--input_h', type=int, default=960,
                                 help='input height')
        self.parser.add_argument('--input_w', type=int, default=1280,
                                 help='input width')

        # train
        self.parser.add_argument('--lr', type=float, default=1.25e-4,
                                 help='learning rate for batch size 32.')
        self.parser.add_argument('--lr_step', type=str, default='90,120',
                                 help='drop learning rate by 10.')
        self.parser.add_argument('--num_epochs', type=int, default=140,
                                 help='total training epochs.')
        self.parser.add_argument('--batch_size', type=int, default=32,
                                 help='total batch size')
        self.parser.add_argument('--val_intervals', type=int, default=10,
                                 help='number of epochs to run validation.')
        self.parser.add_argument('--min_overlap', type=float, default=0.3,
                                 help='iou for train')

        # test
        self.parser.add_argument('--flip_test', action='store_true',
                                 help='flip data augmentation.')
        self.parser.add_argument('--test_scales', type=str, default='1',
                                 help='multi scale test augmentation.')
        self.parser.add_argument('--nms', action='store_true',
                                 help='run nms in testing.')
        self.parser.add_argument('--K', type=int, default=500,
                                 help='max number of output objects.')

        # dataset
        self.parser.add_argument('--shift', type=float, default=0.1,
                                 help='when not using random crop'
                                      'apply shift augmentation.')
        self.parser.add_argument('--scale', type=float, default=0.4,
                                 help='when not using random crop'
                                      'apply scale augmentation.')
        self.parser.add_argument('--rotate', type=float, default=0,
                                 help='when not using random crop'
                                      'apply rotation augmentation.')
        self.parser.add_argument('--flip', type=float, default=0.5,
                                 help='probability of applying flip augmentation.')

        # loss weight
        self.parser.add_argument('--hm_weight', type=float, default=1,
                                 help='loss weight for keypoint heatmaps.')
        self.parser.add_argument('--off_weight', type=float, default=1,
                                 help='loss weight for keypoint local offsets.')
        self.parser.add_argument('--wh_weight', type=float, default=0.1,
                                 help='loss weight for bounding box size.')

    def parse(self):
        opt = self.parser.parse_args()
        os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus

        opt.lr_step = [int(i) for i in opt.lr_step.split(',')]
        opt.test_scales = [float(i) for i in opt.test_scales.split(',')]
        opt.device = torch.device('cuda')

        opt.dataset_info = {
            "num_classes": 10,
            "mean": np.array([0.37294899, 0.37837514, 0.36463863],
                                dtype=np.float32).reshape(1, 1, 3),
            "std": np.array([0.19171683, 0.18299586, 0.19437608],
                            dtype=np.float32).reshape(1, 1, 3),
            "class_name": ['pedestrian', 'people', 'bicycle', 'car',
                            'van', 'truck', 'tricycle', 'awning-tricycle', 'bus', 'motor'],
            "valid_ids": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        }
        
        opt.heads = {'hm': opt.dataset_info["num_classes"], 'wh': 2, 'reg': 2}

        opt.root_dir = os.path.join(os.path.dirname(__file__), '..')
        opt.data_dir = os.path.join(opt.root_dir, 'data', 'visdrone')
        opt.save_dir = os.path.join(opt.root_dir, 'exp', opt.exp_id)
        print('The output will be saved to ', opt.save_dir)
        os.makedirs(opt.save_dir, exist_ok=True)

        return opt


opt = opts().parse()
