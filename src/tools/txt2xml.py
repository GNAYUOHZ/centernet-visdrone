# coding: utf-8
# author: HXY
# 2020-4-17

"""
该脚本用于visdrone数据处理；
将annatations文件夹中的txt标签文件转换为XML文件；
txt标签内容为：
<bbox_left>,<bbox_top>,<bbox_width>,<bbox_height>,<score>,<object_category>,<truncation>,<occlusion>
类别：
ignored regions(0), pedestrian(1),
people(2), bicycle(3), car(4), van(5),
truck(6), tricycle(7), awning-tricycle(8),
bus(9), motor(10), others(11)
"""

import os
import cv2
import time
from xml.dom import minidom
from tqdm import tqdm
name_dict = {'0': 'ignored regions', '1': 'pedestrian', '2': 'people',
             '3': 'bicycle', '4': 'car', '5': 'van', '6': 'truck',
             '7': 'tricycle', '8': 'awning-tricycle', '9': 'bus',
             '10': 'motor', '11': 'others'}


def transfer_to_xml(pic, txt, file_name, xml_save_path):
    img = cv2.imread(pic)
    img_w = img.shape[1]
    img_h = img.shape[0]
    img_d = img.shape[2]
    doc = minidom.Document()

    annotation = doc.createElement("annotation")
    doc.appendChild(annotation)
    folder = doc.createElement('folder')
    folder.appendChild(doc.createTextNode('visdrone'))
    annotation.appendChild(folder)

    filename = doc.createElement('filename')
    filename.appendChild(doc.createTextNode(file_name))
    annotation.appendChild(filename)

    source = doc.createElement('source')
    database = doc.createElement('database')
    database.appendChild(doc.createTextNode("Unknown"))
    source.appendChild(database)

    annotation.appendChild(source)

    size = doc.createElement('size')
    width = doc.createElement('width')
    width.appendChild(doc.createTextNode(str(img_w)))
    size.appendChild(width)
    height = doc.createElement('height')
    height.appendChild(doc.createTextNode(str(img_h)))
    size.appendChild(height)
    depth = doc.createElement('depth')
    depth.appendChild(doc.createTextNode(str(img_d)))
    size.appendChild(depth)
    annotation.appendChild(size)

    segmented = doc.createElement('segmented')
    segmented.appendChild(doc.createTextNode("0"))
    annotation.appendChild(segmented)

    with open(txt, 'r') as f:
        lines = f.readlines()
        for line in lines:
            box = line.strip('\n')
            box = box.split(',')
            x_min = box[0]
            y_min = box[1]
            x_max = int(box[0]) + int(box[2])
            y_max = int(box[1]) + int(box[3])
            object_name = name_dict[box[5]]

            object = doc.createElement('object')
            nm = doc.createElement('name')
            nm.appendChild(doc.createTextNode(object_name))
            object.appendChild(nm)
            pose = doc.createElement('pose')
            pose.appendChild(doc.createTextNode("Unspecified"))
            object.appendChild(pose)
            truncated = doc.createElement('truncated')
            truncated.appendChild(doc.createTextNode("1"))
            object.appendChild(truncated)
            difficult = doc.createElement('difficult')
            difficult.appendChild(doc.createTextNode("0"))
            object.appendChild(difficult)
            bndbox = doc.createElement('bndbox')
            xmin = doc.createElement('xmin')
            xmin.appendChild(doc.createTextNode(x_min))
            bndbox.appendChild(xmin)
            ymin = doc.createElement('ymin')
            ymin.appendChild(doc.createTextNode(y_min))
            bndbox.appendChild(ymin)
            xmax = doc.createElement('xmax')
            xmax.appendChild(doc.createTextNode(str(x_max)))
            bndbox.appendChild(xmax)
            ymax = doc.createElement('ymax')
            ymax.appendChild(doc.createTextNode(str(y_max)))
            bndbox.appendChild(ymax)
            object.appendChild(bndbox)
            annotation.appendChild(object)
            with open(os.path.join(xml_save_path, file_name + '.xml'), 'w') as x:
                x.write(doc.toprettyxml())


if __name__ == '__main__':
    t = time.time()
    print('Transfer .txt to .xml...ing....')
    root_dir = '/home/zy/proj/CenterNet/data/visdrone_org/VisDrone2019-DET-test-dev/'
    txt_folder = root_dir+'annotations'  # visdrone txt标签文件夹
    txt_file = os.listdir(txt_folder)
    img_folder = root_dir+'images'  # visdrone 照片所在文件夹
    xml_save_path = root_dir+'xml'  # 生成的xml文件存储的文件夹
    os.makedirs(xml_save_path, exist_ok=True)

    for i in tqdm(range(len(txt_file))):
        txt = txt_file[i]
        txt_full_path = os.path.join(txt_folder, txt)
        img_full_path = os.path.join(img_folder, txt.split('.')[0] + '.jpg')

        try:
            transfer_to_xml(img_full_path, txt_full_path,
                            txt.split('.')[0], xml_save_path)
        except Exception as e:
            print(e)
  
    print("Transfer .txt to .XML sucessed. costed: {:.3f}s...".format(
        time.time() - t))
