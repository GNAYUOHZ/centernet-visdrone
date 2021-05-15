from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import cv2
from src.opts import opt
from src.detector import Detector
import numpy as np


def save_results(opt, image, results, path):
    for cls_ind in range(1, opt.dataset_info["num_classes"] + 1):
        for bbox in results[cls_ind]:
            conf = bbox[4]
            # filter low score
            if conf < opt.vis_thresh:
                continue
            bbox = np.array(bbox[:4], dtype=np.int32)

            class_name = opt.dataset_info["class_name"]

            cv2.rectangle(img=image,
                          pt1=(bbox[0], bbox[1]),
                          pt2=(bbox[2], bbox[3]),
                          color=[0, 255, 0],
                          thickness=1)
            #txt
            cv2.putText(img=image,
                        text=f'{class_name[cls_ind-1]}{conf:.1f}',
                        org=(bbox[0], bbox[1] - 2),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.5,
                        color=(0, 255, 0),
                        thickness=1,
                        lineType=cv2.LINE_AA)

    cv2.imwrite(path, image)


def demo():
    detector = Detector(opt)
    image_path = opt.image
    print("input image path:", image_path)
    image = cv2.imread(image_path)
    
    ret = detector.run(image)   

    save_results(opt, image, ret['results'], 'demo_result.png')

    time_str = ''
    for stat in ['tot', 'load', 'pre', 'net', 'dec', 'post', 'merge']:
        time_str += f'{stat} {ret[stat]:.3f}s |'
    print(time_str)


if __name__ == '__main__':

    demo()
