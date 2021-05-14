from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
import json
import cv2
import os
import math
import time
from pycocotools import coco
from .utils.image import color_aug, get_affine_transform, affine_transform, gaussian_radius, draw_umich_gaussian


class Dataset(torch.utils.data.Dataset):

    def __init__(self, opt, split):
        super(Dataset, self).__init__()
        self.opt = opt
        self.split = split
        
        self.img_dir = os.path.join(self.opt.data_dir, split)
        self.annot_path = os.path.join(
            self.opt.data_dir, 'annotations', f'instances_{split}.json')
        self.max_objs = 256
        self.class_name = opt.dataset_info["class_name"]
        self.class_nums = len(self.class_name)
        self._valid_ids = opt.dataset_info["valid_ids"]
        self.mean = opt.dataset_info["mean"]
        self.std = opt.dataset_info["std"]
        self.cat_ids = {v: i for i, v in enumerate(self._valid_ids)}
        # print(self.cat_ids) # {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 7, 9: 8, 10: 9}
        self._data_rng = np.random.RandomState(123)
        self._eig_val = np.array([0.2141788, 0.01817699, 0.00341571],
                                 dtype=np.float32)
        self._eig_vec = np.array([
            [-0.58752847, -0.69563484, 0.41340352],
            [-0.5832747, 0.00994535, -0.81221408],
            [-0.56089297, 0.71832671, 0.41158938]
        ], dtype=np.float32)

        print(f'==> initializing visdrone {split} data.')
        self.coco = coco.COCO(self.annot_path)
        self.images = self.coco.getImgIds()
        print(f'Loaded {split} {len(self.images)} samples')

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_id = self.images[index]

        file_name = self.coco.loadImgs(ids=[img_id])[0]['file_name']
        img = cv2.imread(os.path.join(self.img_dir, file_name))

        ann_ids = self.coco.getAnnIds(imgIds=[img_id])
        anns = self.coco.loadAnns(ids=ann_ids)

        # print('test1.png',img.shape,img.shape[1]/img.shape[0])
        # cv2.imwrite('test1.png',img)

        height, width = img.shape[0], img.shape[1]
        center = np.array([width / 2., height / 2.], dtype=np.float32)

        input_h = self.opt.input_h
        input_w = self.opt.input_w
        output_h = input_h // 2 # down ratio = 2
        output_w = input_w // 2
        scale = max(height, width) * 1.0
        rot = 0
        flipped = False
        if self.split == 'train':
            # clip 限制在min, max
            cf = self.opt.shift
            center[0] += scale * np.clip(np.random.randn()*cf, -2*cf, 2*cf)
            center[1] += scale * np.clip(np.random.randn()*cf, -2*cf, 2*cf)
          
            sf = self.opt.scale
            scale += scale * np.clip(np.random.randn()*sf, - sf, sf)
     
            rf = self.opt.rotate
            rot = np.clip(np.random.randn()*rf, - rf, rf)
        
            if np.random.random() < self.opt.flip:
                flipped = True
                img = img[:, ::-1, :]
                center[0] = width - center[0] - 1 
            
        trans_input = get_affine_transform(center=center,
                                           scale=scale,
                                           rot=rot,
                                           output_size=[input_w, input_h])

        trans_output = get_affine_transform(center=center,
                                            scale=scale,
                                            rot=rot,
                                            output_size=[output_w, output_h])

        inp = cv2.warpAffine(src=img,
                             M=trans_input,  
                             dsize=(input_w, input_h),  
                             flags=cv2.INTER_LINEAR) 
       
        # cv2.imwrite('gs_mask_ori.png', inp)
        # time.sleep(5)
        # x=inp

        inp = (inp.astype(np.float32) / 255.)
        if self.split == 'train':
            color_aug(self._data_rng, inp, self._eig_val, self._eig_vec)

        inp = (inp - self.mean) / self.std
        inp = inp.transpose(2, 0, 1)  # from [H, W, C] to [C, H, W]

        hm = np.zeros((self.class_nums, output_h, output_w), dtype=np.float32)
        wh = np.zeros((self.max_objs, 2), dtype=np.float32)  # width and height
        reg = np.zeros((self.max_objs, 2), dtype=np.float32)  # regression
        ind = np.zeros((self.max_objs), dtype=np.int64)  # indexs
        reg_mask = np.zeros((self.max_objs), dtype=np.uint8)  # choose which index

        for k in range(min(len(anns), self.max_objs)):
            ann = anns[k]
            box = ann['bbox']

            # 去除ignore和other
            if ann['category_id'] not in self.cat_ids.keys():
                continue
            bbox = np.array([box[0], box[1], box[0] + box[2], box[1] + box[3]],
                            dtype=np.float32)  # xyxy

            if flipped:
                bbox[[0, 2]] = width - bbox[[2, 0]] - 1

            bbox[:2] = affine_transform(bbox[:2], trans_output)
            bbox[2:] = affine_transform(bbox[2:], trans_output)

            # prevant overflow
            bbox[[0, 2]] = np.clip(bbox[[0, 2]], 0, output_w - 1)
            bbox[[1, 3]] = np.clip(bbox[[1, 3]], 0, output_h - 1)

            # cv2.rectangle(img=x,
            #               pt1=(int(bbox[0]*4), int(bbox[1]*4)),
            #               pt2=(int(bbox[2]*4), int(bbox[3]*4)),
            #               color=[0, 255, 0],
            #               thickness=2)

            h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]

            if h > 0 and w > 0:
                ct = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2],
                              dtype=np.float32)  # float
                ct_int = ct.astype(np.int32)  # int

                cls_id = self.cat_ids[ann['category_id']]
                
                # 根据一元二次方程计算出最小的半径
                
                radius = gaussian_radius(
                    (math.ceil(h), math.ceil(w)), min_overlap = self.opt.min_overlap)
                radius = max(0, int(radius))
                draw_umich_gaussian(hm[cls_id], ct_int, radius)
         
                # 长宽
                wh[k] = 1. * w, 1. * h
                # 当前是 obj 序列中的第 k 个 = fmap_w * cy + cx = fmap 中的序列数
                ind[k] = ct_int[1] * output_w + ct_int[0]
                # 记录偏移量
                reg[k] = ct - ct_int  # discretization error
                # 进行 mask 标记
                reg_mask[k] = 1
        
        # time.sleep(10)

        ret = {'input': inp,
               'hm': hm,
               'wh': wh,
               'reg': reg,
               'reg_mask': reg_mask,
               'ind': ind,
               }

        gs_mask = np.zeros((output_h, output_w))
        for i in range(0, 10):
            np.maximum(gs_mask, hm[i], out=gs_mask)

        # cv2.imwrite("gs_mask.jpg", gs_mask * 255)
      
        # time.sleep(10)

        return ret


if __name__ == "__main__":
    from opts import opt

    train_loader = torch.utils.data.DataLoader(
        Dataset(opt, 'train'),
        batch_size=1,
        shuffle=True,
        num_workers=1,
        pin_memory=True,
        drop_last=True
    )
    for iter_id, batch in enumerate(train_loader):

        if iter_id % 100 == 0:
            print(iter_id)
