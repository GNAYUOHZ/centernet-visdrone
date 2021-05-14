import os
import glob
import json
import shutil
import numpy as np
import xml.etree.ElementTree as ET


def get_and_check(root, name, length):
    vars = root.findall(name)
    if len(vars) == 0:
        raise NotImplementedError('Can not find %s in %s.' % (name, root.tag))
    if length and len(vars) != length:
        raise NotImplementedError(
            'The size of %s is supposed to be %d, but is %d.' % (name, length, len(vars)))
    if length == 1:
        vars = vars[0]
    return vars


def convert(xml_list, json_file):
    json_dict = {"images": [], "type": "instances",
                 "annotations": [], "categories": []}
    categories = pre_define_categories.copy()
    bnd_id = 1  # START_BOUNDING_BOX_ID
    for index, line in enumerate(xml_list):
        print("Processing %s" % (line))
        xml_f = line
        tree = ET.parse(xml_f)
        root = tree.getroot()

        filename = os.path.basename(xml_f)[:-4] + ".jpg"
        image_id = index
        size = get_and_check(root, 'size', 1)
        width = int(get_and_check(size, 'width', 1).text)
        height = int(get_and_check(size, 'height', 1).text)
        image = {'file_name': filename, 'height': height,
                 'width': width, 'id': image_id}
        json_dict['images'].append(image)
        #  Currently we do not support segmentation
        segmented = get_and_check(root, 'segmented', 1).text
        assert segmented == '0'
        for obj in root.findall('object'):
            category = get_and_check(obj, 'name', 1).text
            category_id = categories[category]
            # # truncation
            # truncation = get_and_check(obj, 'truncation', 1)
            # # occlusion
            # occlusion = get_and_check(obj, 'occlusion', 1)
            # bbox
            bndbox = get_and_check(obj, 'bndbox', 1)
            xmin = int(float(get_and_check(bndbox, 'xmin', 1).text))
            ymin = int(float(get_and_check(bndbox, 'ymin', 1).text))
            xmax = int(float(get_and_check(bndbox, 'xmax', 1).text))
            ymax = int(float(get_and_check(bndbox, 'ymax', 1).text))
            assert (xmax > xmin), "xmax <= xmin, {}".format(line)
            if ymax <= ymin:
                print(f"{ymax} <= {ymin}, {line}")
                continue
            o_width = abs(xmax - xmin)
            o_height = abs(ymax - ymin)

            ann = {'area': o_width * o_height, 'iscrowd': 0, 'image_id':
                   image_id, 'bbox': [xmin, ymin, o_width, o_height],
                   'category_id': category_id, 'id': bnd_id, 'ignore': 0,
                   'segmentation': []}
            json_dict['annotations'].append(ann)
            bnd_id = bnd_id + 1

    for cate, cid in categories.items():
        cat = {'supercategory': 'all', 'id': cid, 'name': cate}
        json_dict['categories'].append(cat)
    json_fp = open(json_file, 'w')
    json_str = json.dumps(json_dict)
    json_fp.write(json_str)
    json_fp.close()
    print("------------create {} done--------------".format(json_file))


if __name__ == '__main__':
    pre_define_categories = {'ignored regions': 0,
                             'pedestrian': 1,
                             'people': 2,
                             'bicycle': 3,
                             'car': 4,
                             'van': 5,
                             'truck': 6,
                             'tricycle': 7,
                             'awning-tricycle': 8,
                             'bus': 9,
                             'motor': 10,
                             'others': 11}

    # 保存的json文件
    root_dir = '/home/zy/proj/CenterNet/data/visdrone_org/VisDrone2019-DET-train/'
    save_json = root_dir+'instances_train.json'

    # 初始文件所在的路径
    xml_dir = root_dir+"xml"
    xml_list = glob.glob(xml_dir + "/*.xml")
    convert(xml_list, save_json)

