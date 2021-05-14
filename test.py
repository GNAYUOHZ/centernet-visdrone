from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import json
import cv2
import numpy as np
import time
import torch

from src.opts import opt
from src.utils.average_meter import AverageMeter
from src.detector import Detector
from src.dataset import Dataset
from src.utils.logger import Logger
from src.tools.viseval.eval_det import eval_det


def test_visdrone():
    torch.manual_seed(317)
    logger = Logger(opt, "test")
    dataset = Dataset(opt, "val")
    detector = Detector(opt)

    time_stats = ['tot', 'load', 'pre', 'net', 'dec', 'post', 'merge']
    avg_time_stats = {t: AverageMeter() for t in time_stats}

    all_gt = []
    all_det = []
    allheight = []
    allwidth = []

    detections = []
    for iter_id in range(len(dataset)):
        img_id = dataset.images[iter_id]
        img_info = dataset.coco.loadImgs(ids=[img_id])[0]
        img_path = os.path.join(dataset.img_dir, img_info['file_name'])
        image = cv2.imread(img_path)

        ret = detector.run(image)

        # det
        # convert_eval_format
        det = []
        for cls_ind, bboxs in ret['results'].items():
            category_id = dataset._valid_ids[cls_ind - 1]
            for bbox in bboxs:
                bbox[2] -= bbox[0]
                bbox[3] -= bbox[1]
                for i in range(4):
                    bbox[i] = round(bbox[i], 2)
                score = round(bbox[4], 2)
                # coco
                detections.append({
                    "image_id": int(img_id),
                    "category_id": int(category_id),
                    "bbox": bbox[0:4],
                    "score": score,
                })

                det.append([bbox[0], bbox[1], bbox[2], bbox[3],
                            score, category_id, -1, -1])

                # f.write(f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]},{score},{category_id},{-1},{-1}\n") 
        det = np.array(det)

        # gt
        label = []
        ann_ids = dataset.coco.getAnnIds(imgIds=[iter_id])
        anns = dataset.coco.loadAnns(ids=ann_ids)
        for ann in anns:
            bbox = ann['bbox']
            category_id = ann['category_id']

            score = 0 if category_id == 0 or category_id == 11 else 1
            label.append([bbox[0], bbox[1], bbox[2], bbox[3],
                          score, category_id, -1, -1])

        label = np.array(label)

        height, width = image.shape[:2]

        allheight.append(height)
        allwidth.append(width)
        all_det.append(det)
        all_gt.append(label)

        info = f'[{iter_id}/{len(dataset)}]'
        for t in avg_time_stats:
            avg_time_stats[t].update(ret[t])
            info += '|{} {tm.val:.3f}s ({tm.avg:.3f}s) '.format(
                t, tm=avg_time_stats[t])
        # log
        if iter_id % 50 == 0:
            logger.write(info)

    # visdrone eval
    ap_all, ap_50, ap_75, ar_1, ar_10, ar_100, ar_500, ap_classwise = eval_det(
        all_gt, all_det, allheight, allwidth, per_class=True)

    logger.write(f'AP [IoU=0.50:0.95 | maxDets=500] = {ap_all:3.2f}%.')
    logger.write(f'AP [IoU=0.50      | maxDets=500] = {ap_50:3.2f}%.')
    logger.write(f'AP [IoU=0.75      | maxDets=500] = {ap_75:3.2f}%.')
    logger.write(f'AR [IoU=0.50:0.95 | maxDets=  1] = {ar_1:3.2f}%.')
    logger.write(f'AR [IoU=0.50:0.95 | maxDets= 10] = {ar_10:3.2f}%.')
    logger.write(f'AR [IoU=0.50:0.95 | maxDets=100] = {ar_100:3.2f}%.')
    logger.write(f'AR [IoU=0.50:0.95 | maxDets=500] = {ar_500:3.2f}%.')

    for i, ap in enumerate(ap_classwise):
        logger.write(
            f'Class {opt.dataset_info["class_name"][i]:15} AP = {ap:3.2f}%')

    # from pycocotools.cocoeval import COCOeval
    # coco eval
    # result_json = os.path.join(save_dir, "results.json")
    # json.dump(detections, open(result_json, "w"))
    # dataset.coco.dataset['categories'].pop(0)
    # dataset.coco.dataset['categories'].pop(-1)

    # coco_dets = dataset.coco.loadRes(detections)
    # coco_eval = COCOeval(dataset.coco, coco_dets, "bbox")
    # coco_eval.evaluate()
    # coco_eval.accumulate()
    # ret = coco_eval.summarize()
    # logger.write(ret)


if __name__ == '__main__':
    test_visdrone()
