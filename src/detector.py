from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import os
import numpy as np
import time
import torch
from torchvision.ops import nms
from .utils.decode import ctdet_decode
from .utils.image import get_affine_transform, affine_transform
from .utils.soft_nms import soft_nms
from .model import get_model, load_model, save_model


class Detector():
    def __init__(self, opt):
        print('Creating model...')
        self.model = get_model(opt.arch, opt.heads)
        self.model = load_model(self.model, opt.load_model)
        self.model = self.model.to(opt.device)
        self.model.eval()

        self.mean = opt.dataset_info["mean"]
        self.std = opt.dataset_info["std"]
        self.num_classes = opt.dataset_info["num_classes"]
        self.opt = opt

    def pre_process(self, image, scale):
        height, width = image.shape[0:2]
        new_height = int(height * scale)
        new_width = int(width * scale)
        center = np.array([new_width / 2., new_height / 2.], dtype=np.float32)

        input_h = self.opt.input_h
        input_w = self.opt.input_w
        scale = max(height, width) * 1.0

        # 对 Ground Truth heatmap 进行仿射变换
        trans_input = get_affine_transform(center=center,
                                           scale=scale,
                                           rot=0,
                                           output_size=[input_w, input_h])
        resized_image = cv2.resize(image, (new_width, new_height))
        inp = cv2.warpAffine(src=resized_image,
                             M=trans_input,  # 变换矩阵
                             dsize=(input_w, input_h),  # 输出图像的大小
                             flags=cv2.INTER_LINEAR)  # 插值方法的组合

        inp = (inp.astype(np.float32) / 255.)
        inp = (inp - self.mean) / self.std

        images = inp.transpose(2, 0, 1).reshape(1, 3, input_h, input_w)
        if self.opt.flip_test:
            images = np.concatenate((images, images[:, :, :, ::-1]), axis=0)

        images = torch.from_numpy(images).to(self.opt.device)
        meta = {'c': center,
                's': scale,
                'out_height': input_h // 2,
                'out_width': input_w // 2}
        return images, meta

    def process(self, images):
        with torch.no_grad():
            output = self.model(images)

            hm = output['hm'].sigmoid_()
            wh = output['wh']
            reg = output['reg']
            if self.opt.flip_test:
                # flip_tensor
                hm = (hm[0:1] + torch.flip(hm[1:2], [3])) / 2
                wh = (wh[0:1] + torch.flip(wh[1:2], [3])) / 2
                reg = reg[0:1]
            torch.cuda.synchronize()
            forward_time = time.time()
            dets = ctdet_decode(hm, wh, reg, K=self.opt.K)
        return output, dets, forward_time

    def post_process(self, dets, meta, scale=1):
        dets = dets.detach().cpu().numpy()[0]

        trans = get_affine_transform(
            meta['c'], meta['s'], 0, (meta['out_width'], meta['out_height']), inv=1)

        top_preds = {}
        for p in range(dets.shape[0]):
            dets[p, 0:2] = affine_transform(dets[p, 0:2], trans)
            dets[p, 2:4] = affine_transform(dets[p, 2:4], trans)
        classes = dets[:, -1]
        for j in range(self.num_classes):
            inds = (classes == j)
            top_preds[j + 1] = np.concatenate([
                dets[inds, :4].astype(np.float32),
                dets[inds, 4:5].astype(np.float32)], axis=1).reshape(-1, 5)
            top_preds[j + 1][:, :4] /= scale
        return top_preds

    def merge_outputs(self, detections):
        results = {}
        for j in range(1, self.num_classes + 1):
            results[j] = np.concatenate(
                [detection[j] for detection in detections], axis=0).astype(np.float32)
            if len(self.opt.test_scales) > 1 or self.opt.nms:
                results[j]=soft_nms(results[j], Nt=0.5, method=2, threshold=0.3)

        scores = np.hstack(
            [results[j][:, 4] for j in range(1, self.num_classes + 1)])
        if len(scores) > self.opt.K:
            kth = len(scores) - self.opt.K
            thresh = np.partition(scores, kth)[kth]
            for j in range(1, self.num_classes + 1):
                keep_inds = (results[j][:, 4] >= thresh)
                results[j] = results[j][keep_inds]
        return results

    def run(self, image):
        load_time, pre_time, net_time, dec_time, post_time, merge_time, tot_time = 0, 0, 0, 0, 0, 0, 0

        start_time = time.time()
        # 加载，忽略
        loaded_time = time.time()
        load_time += (loaded_time - start_time)

        detections = []
        for scale in self.opt.test_scales:
            scale_start_time = time.time()
            # 预处理
            images, meta = self.pre_process(image, scale)
            # print(images.shape)
            torch.cuda.synchronize()
            pre_process_time = time.time()
            pre_time += pre_process_time - scale_start_time

            # 模型推理和解码
            output, dets, forward_time = self.process(images)
            torch.cuda.synchronize()
            net_time += forward_time - pre_process_time
            decode_time = time.time()
            dec_time += decode_time - forward_time

            # 后处理
            dets = self.post_process(dets, meta, scale)
            torch.cuda.synchronize()
            post_process_time = time.time()
            post_time += post_process_time - decode_time

            detections.append(dets)

        # 合并结果
        results = self.merge_outputs(detections)
        torch.cuda.synchronize()
        end_time = time.time()
        merge_time += end_time - post_process_time
        tot_time += end_time - start_time

        return {'results': results,
                'tot': tot_time,
                'load': load_time,
                'pre': pre_time,
                'net': net_time,
                'dec': dec_time,
                'post': post_time,
                'merge': merge_time
                }
