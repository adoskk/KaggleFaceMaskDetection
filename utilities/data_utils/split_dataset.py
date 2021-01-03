import os
from xml.dom.minidom import parse
from shutil import copyfile
import random
import re


print('total image num = ', len(os.listdir(os.path.join('../../data/original_data', "images"))))
wo_num = 0
w_num = 0
wo_image_num = 0
w_image_num = 0
for dirname, _, filenames in os.walk('../../data/original_data/annotations'):
    for filename in filenames:
        dom = parse(os.path.join('../../data/original_data/annotations', filename))
        root = dom.documentElement
        objects = root.getElementsByTagName("object")
        for o in objects:
            label_type = o.getElementsByTagName("name")[0].childNodes[0].data
            if label_type == 'without_mask':
                wo_num += 1
            else:
                w_num += 1
        w_image_num += 1
        for o in objects:
            label_type = o.getElementsByTagName("name")[0].childNodes[0].data
            if label_type == 'without_mask':
                wo_image_num += 1
                break
print('total without mask object: ', wo_num)
print('total with mask object: ', w_num)
print('total images without mask object: ', wo_image_num)
print('total images with mask object: ', w_image_num)

if not os.path.exists('../../data/train'):
    os.mkdir('../../data/train')
    os.mkdir('../../data/train/images')
    os.mkdir('../../data/train/annotations')
    os.mkdir('../../data/test')
    os.mkdir('../../data/test/images')
    os.mkdir('../../data/test/annotations')
annotation_list = os.listdir('../../data/original_data/annotations')

random.seed(10)
random.shuffle(annotation_list)
train_list = annotation_list[:int(len(annotation_list)/4*3)]
test_list = annotation_list[int(len(annotation_list)/4*3):]

train_num = 0
for filename in train_list:
    img_id = int(re.findall(r'\d+', filename)[0])
    image_name = '../../data/original_data/images/maksssksksss' + str(img_id)+'.png'
    dom = parse(os.path.join('../../data/original_data/annotations', filename))
    root = dom.documentElement
    objects = root.getElementsByTagName("object")
    wo_mask = False
    for o in objects:
        label_type = o.getElementsByTagName("name")[0].childNodes[0].data
        if label_type == 'without_mask':
            wo_mask = True
            break
    if wo_mask:
        for ii in range(4):
            copyfile(image_name, '../../data/train/images/maksssksksss' + str(train_num)+'.png')
            copyfile(os.path.join('../../data/original_data/annotations', filename), \
                     '../../data/train/annotations/maksssksksss' + str(train_num)+'.xml')
            train_num += 1
    else:
        copyfile(image_name, '../../data/train/images/maksssksksss' + str(train_num) + '.png')
        copyfile(os.path.join('../../data/original_data/annotations', filename), \
                 '../../data/train/annotations/maksssksksss' + str(train_num) + '.xml')
        train_num += 1
        
test_num = 0
for filename in test_list:
    img_id = int(re.findall(r'\d+', filename)[0])
    image_name = '../../data/original_data/images/maksssksksss' + str(img_id)+'.png'
    dom = parse(os.path.join('../../data/original_data/annotations', filename))
    root = dom.documentElement
    objects = root.getElementsByTagName("object")

    copyfile(image_name, '../../data/test/images/maksssksksss' + str(test_num) + '.png')
    copyfile(os.path.join('../../data/original_data/annotations', filename), \
             '../../data/test/annotations/maksssksksss' + str(test_num) + '.xml')
    test_num += 1
    

print('total training num: ', train_num)
print('total testing num: ', test_num)