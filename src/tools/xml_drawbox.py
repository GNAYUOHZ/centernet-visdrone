# 第二步，检验一下所转换的xml格式画回原图中是否准确
import os
import os.path
import numpy as np
import xml.etree.ElementTree as xmlET
from PIL import Image, ImageDraw

classes = ('__background__', # always index 0
           'ignored regions','pedestrian','people','bicycle','car','van','truck','tricycle','awning-tricycle','bus','motor','others')


file_path_img = '/home/zy/proj/CenterNet/data/visdrone_org/VisDrone2019-DET-val/images'
file_path_xml = '/home/zy/proj/CenterNet/data/visdrone_org/VisDrone2019-DET-val/xml'
save_file_path = '/home/zy/proj/CenterNet/data/visdrone_org/VisDrone2019-DET-val/images_box'
os.makedirs(save_file_path,exist_ok=True)

pathDir = os.listdir(file_path_xml)
count_map={}
for idx in range(len(pathDir)):  
    print(idx)
    filename = pathDir[idx]
    tree = xmlET.parse(os.path.join(file_path_xml, filename))
    objs = tree.findall('object')        
    num_objs = len(objs)
    boxes = np.zeros((num_objs, 5), dtype=np.uint16)

    for ix, obj in enumerate(objs):
        
        bbox = obj.find('bndbox')
        # Make pixel indexes 0-based
        x1 = float(bbox.find('xmin').text) 
        y1 = float(bbox.find('ymin').text) 
        x2 = float(bbox.find('xmax').text) 
        y2 = float(bbox.find('ymax').text) 

        cla = obj.find('name').text
        label = classes.index(cla)
       
        boxes[ix, 0:4] = [x1, y1, x2, y2]
        boxes[ix, 4] = label
        # count
        if classes[label] not in count_map:
            count_map[classes[label]]=1
        else:
            count_map[classes[label]]+=1

    image_name = os.path.splitext(filename)[0]
    img = Image.open(os.path.join(file_path_img, image_name + '.jpg'))

    draw = ImageDraw.Draw(img)
    for ix in range(len(boxes)):
        xmin = int(boxes[ix, 0])
        ymin = int(boxes[ix, 1])
        xmax = int(boxes[ix, 2])
        ymax = int(boxes[ix, 3])
        draw.rectangle([xmin, ymin, xmax, ymax], outline=(255, 0, 0))
        draw.text([xmin, ymin], classes[boxes[ix, 4]], (255, 0, 0))

    img.save(os.path.join(save_file_path, image_name + '.png'))
print(count_map)