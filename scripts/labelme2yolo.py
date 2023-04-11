import os
import shutil

import cv2
import json
import numpy as np
import glob


def classes_yaml_print():
    c = '''huashang
    cashang
    pengshang
    yise
    aokeng
    yashang
    gubao
    bianxing'''

    c = c.split('\n')
    for i, v, in enumerate(c):
        print(f"{i}: {v}")


def find_no_seg_images(data_root):
    count = 0
    labels = glob.glob(f"{data_root}/labels/*/*.txt")
    for label_txt in labels:
        with open(label_txt, "r") as f:
            lines = f.readlines()

            for info in lines:
                info_splits = info.split(" ")
                if len(info_splits) <= 5:
                    count += 1
                    print(f"not segment, remove: {label_txt}")
                    img_path = label_txt.replace("labels", "images").replace(".txt", ".jpg")
                    os.remove(label_txt)
                    os.remove(img_path)
                    break
    print(f"count: {count}")


def labelme_seg_to_yolo(json_path, save_path, img_shape, classes):
    h, w = img_shape[0], img_shape[1]
    with open(json_path, 'r') as f:
        masks = json.load(f)['shapes']
    with open(save_path, 'w+') as f:
        for idx, mask_data in enumerate(masks):
            mask_label = mask_data['label']
            mask = np.array([np.array(i) for i in mask_data['points']], dtype=float)
            mask[:, 0] /= w
            mask[:, 1] /= h
            mask = mask.reshape((-1))
            if idx != 0:
                f.write('\n')
            f.write(f'{classes.index(mask_label)} {" ".join(list(map(lambda x: f"{x:.6f}", mask)))}')


def convert_labelme_jsons_to_yolo(img_path_list, save_dir, classes, mode, contact_dir_name=True):
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    imgs_save_dir = os.path.join(save_dir, "images")
    if not os.path.exists(imgs_save_dir):
        os.mkdir(imgs_save_dir)
    labels_save_dir = os.path.join(save_dir, "labels")
    if not os.path.exists(labels_save_dir):
        os.mkdir(labels_save_dir)

    for img_path in img_path_list:
        img_base_name = os.path.basename(img_path)
        img_pure_name = os.path.splitext(img_base_name)[0]
        img_dir_name = os.path.basename(os.path.dirname(img_path))
        json_path = img_path.replace(".jpg", ".json")
        if not os.path.exists(json_path):
            continue

        img_save_dir_path = os.path.join(imgs_save_dir, mode) if contact_dir_name else os.path.join(imgs_save_dir, mode,
                                                                                                    img_dir_name)
        if not os.path.exists(img_save_dir_path):
            os.makedirs(img_save_dir_path, exist_ok=True)
        img_save_path = os.path.join(img_save_dir_path, f"{img_dir_name}_M_{img_pure_name}.jpg") if contact_dir_name else\
            os.path.join(img_save_dir_path, img_base_name)
        shutil.copyfile(img_path, img_save_path)
        image = cv2.imread(img_save_path)

        txt_save_dir_path = os.path.join(labels_save_dir, mode) if contact_dir_name else os.path.join(labels_save_dir,
                                                                                                      mode,
                                                                                                      img_dir_name)
        if not os.path.exists(txt_save_dir_path):
            os.makedirs(txt_save_dir_path, exist_ok=True)
        txt_save_path = os.path.join(txt_save_dir_path, f"{img_dir_name}_M_{img_pure_name}.txt") if contact_dir_name \
            else os.path.join(txt_save_dir_path, img_base_name.replace(".jpg", ".txt"))
        labelme_seg_to_yolo(json_path, txt_save_path, image.shape, classes)


def train_val_data_to_yolo_data(data_root, save_dir, class_name_list_file):
    with open(class_name_list_file, 'r') as f:
        classes = f.readlines()
        classes = [i.strip() for i in classes]
    train_data = os.path.join(data_root, "train")

    train_imgs = glob.glob(f"{train_data}/*/*.jpg")
    convert_labelme_jsons_to_yolo(train_imgs, save_dir, classes, "train")

    val_data = os.path.join(data_root, "val")

    val_imgs = glob.glob(f"{val_data}/*/*.jpg")
    convert_labelme_jsons_to_yolo(val_imgs, save_dir, classes, "val")


if __name__ == '__main__':
    # train_val_data_to_yolo_data("/data/Data2Model/huawei_pc_2023-03-29_clear_with_ok/train_val_data_cropped",
    #                             "/data/Data2Model/huawei_pc_2023-03-29_clear_with_ok/yolo_data",
    #                             "/data/Data2Model/huawei_pc_2023-03-29_clear_with_ok/class_names_list.txt")

    find_no_seg_images("/data/Data2Model/huawei_pc_2023-03-29_clear_with_ok/yolo_data")

