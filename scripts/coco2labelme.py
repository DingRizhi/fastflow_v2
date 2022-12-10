from __future__ import print_function
import os
import pylab
import json
from pycocotools.coco import COCO
from base64 import b64encode
import shutil
import cv2


def labelme_shapes(annotations, categories, class_list=None):
    shapes = []
    # label_num = {'person': 0, 'bicycle': 0, 'car': 0, 'motorcycle': 0, 'bus': 0, 'train': 0, 'truck': 0}  # 根据你的数据来修改
    for ann in annotations:
        shape = {}
        class_name = [i['name'] for i in categories if i['id'] == ann['category_id']][0]

        if class_list is not None and len(class_list) > 0 and class_name not in class_list:
            continue
        # label要对应每一类从_1开始编号
        # label_num[class_name[0]] += 1
        # shape['label'] = class_name[0] + '_' + str(label_num[class_name[0]])
        shape['label'] = class_name
        # shape['line_color'] = data_ref['shapes'][0]['line_color']
        # shape['fill_color'] = data_ref['shapes'][0]['fill_color']

        bbox = ann['bbox']
        shape['points'] = [[bbox[0], bbox[1]], [bbox[0] + bbox[2], bbox[1] + bbox[3]]]

        shape['shape_type'] = 'rectangle'
        shape['flags'] = {}
        shape['group_id'] = None

        shapes.append(shape)

    return shapes


def convert_coco_json_file_to_labelme_json(coco_json, image_dir):
    json_file = coco_json  # # Object Instance 类型的标注
    # json_file='./annotations/person_keypoints_val2017.json'  # Object Keypoint 类型的标注格式
    # json_file='./annotations/captions_val2017.json' # Image Caption的标注格式
    coco = COCO(json_file)

    with open(json_file, 'r') as f:
        data = json.load(f)

        images = data['images']
        annotations = data['annotations']
        categories = data['categories']

        for image_info in images:

            img_id = image_info['id']

            img_labelme = {"version": "5.0.5", "flags": {}}
            labelme_annotation = []
            for ann in annotations:
                if ann['image_id'] == img_id:
                    labelme_annotation.append(ann)

            labelme_shape = labelme_shapes(labelme_annotation, categories)
            img_labelme["shapes"] = labelme_shape
            file_name = image_info["file_name"]
            img_labelme['imagePath'] = file_name
            img_labelme['imageData'] = None
            # print(img_labelme)

            json_path = os.path.join(image_dir, file_name.replace('jpg', 'json'))
            with open(json_path, 'w') as f:
                json.dump(img_labelme, f, indent=2)
                print(f"dump json: {json_path}")


def extract_specific_classes_images_and_labelme_json_in_coco(coco_json, image_dir, save_dir, class_list=["person"]):
    json_file = coco_json  # # Object Instance 类型的标注
    # json_file='./annotations/person_keypoints_val2017.json'  # Object Keypoint 类型的标注格式
    # json_file='./annotations/captions_val2017.json' # Image Caption的标注格式
    coco = COCO(json_file)

    with open(json_file, 'r') as f:
        data = json.load(f)

        images = data['images']
        annotations = data['annotations']
        categories = data['categories']

        for image_info in images:

            img_id = image_info['id']

            img_labelme = {"version": "5.0.5", "flags": {}}
            labelme_annotation = []
            for ann in annotations:
                if ann['image_id'] == img_id:
                    labelme_annotation.append(ann)

            labelme_shape = labelme_shapes(labelme_annotation, categories, class_list)
            if len(labelme_shape) == 0:
                continue
            img_labelme["shapes"] = labelme_shape
            file_name = image_info["file_name"]
            img_labelme['imagePath'] = file_name
            # 读取二进制图片，获得原始字节码
            with open(os.path.join(image_dir, file_name), 'rb') as jpg_file:
                byte_content = jpg_file.read()
            img = cv2.imread(os.path.join(image_dir, file_name))
            h, w, c = img.shape
            # 把原始字节码编码成base64字节码
            base64_bytes = b64encode(byte_content)
            # 把base64字节码解码成utf-8格式的字符串
            base64_string = base64_bytes.decode('utf-8')
            # 用字典的形式保存数据
            img_labelme["imageData"] = base64_string
            img_labelme["imageHeight"] = h
            img_labelme["imageWidth"] = w

            json_path = os.path.join(save_dir, file_name.replace('jpg', 'json'))
            with open(json_path, 'w') as f:
                json.dump(img_labelme, f, indent=2)
                shutil.copyfile(os.path.join(image_dir, file_name), os.path.join(save_dir, file_name))
                print(f"dump json: {json_path}")


if __name__ == '__main__':
    extract_specific_classes_images_and_labelme_json_in_coco(
        "/home/log/PycharmProjects/new-YOLOv1_PyTorch/data/scripts/COCO/annotations/instances_train2017.json",
        "/home/log/PycharmProjects/new-YOLOv1_PyTorch/data/scripts/COCO/train2017",
        "/home/log/PycharmProjects/new-YOLOv1_PyTorch/data/scripts/COCO/coco_classes/person/train2017")
