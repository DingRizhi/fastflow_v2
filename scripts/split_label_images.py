import os
import glob
import json
import copy
import shutil
from PIL import Image


def split_label_crop_images(labeled_img_dir, save_dir):
    img_path_list = glob.glob(f"{labeled_img_dir}/*.jpg")

    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    for img_path in img_path_list:
        json_path = img_path.replace("jpg", "json")
        image = Image.open(img_path)
        image_pure_name = os.path.splitext(os.path.basename(img_path))[0]
        if not os.path.exists(json_path):
            print(f"not exists json path: {json_path}")
            continue

        with open(json_path, 'r') as f:
            label_info = json.load(f)

            shapes = label_info["shapes"]
            for index, shape in enumerate(shapes):
                label_name = shape["label"]
                points = shape["points"]
                x1, y1, x2, y2 = 65535, 65535, -65535, -65535
                for point in points:
                    x1, y1, x2, y2 = (
                        min(x1, point[0]),
                        min(y1, point[1]),
                        max(x2, point[0]),
                        max(y2, point[1])
                    )
                print(x1, y1, x2, y2)
                img_ = image.crop((x1, y1, x2, y2))
                label_save_dir = os.path.join(save_dir, label_name)
                if not os.path.exists(label_save_dir):
                    os.makedirs(label_save_dir, exist_ok=True)

                img_.save(os.path.join(label_save_dir, f"{image_pure_name}_{index}.jpg"))



def split_label_original_images(labeled_img_dir, save_dir):
    img_path_list = glob.glob(f"{labeled_img_dir}/*.jpg")

    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    for img_path in img_path_list:
        json_path = img_path.replace("jpg", "json")
        # image = Image.open(img_path)
        image_base_name = os.path.basename(img_path)
        image_pure_name = os.path.splitext(image_base_name)[0]
        if not os.path.exists(json_path):
            print(f"not exists json path: {json_path}")
            continue

        with open(json_path, 'r') as f:
            label_info = json.load(f)

            shapes = label_info["shapes"]
            for index, shape in enumerate(shapes):
                label_name = shape["label"]
                label_save_dir = os.path.join(save_dir, label_name)
                if not os.path.exists(label_save_dir):
                    os.makedirs(label_save_dir, exist_ok=True)

                save_image_path = os.path.join(label_save_dir, image_base_name)
                if os.path.exists(save_image_path):
                    continue
                shutil.copyfile(img_path, save_image_path)
                shutil.copyfile(json_path, os.path.join(label_save_dir, image_base_name.replace("jpg", "json")))


if __name__ == '__main__':
    # split_label_crop_images("/home/log/Downloads/0106-伤线过杀残胶数据-19pcs", "/data/BYD_dingzi/canjiao_crop_0106")
    split_label_original_images("/data/BYD_dingzi/detection_data/01_sx_zp_nqp_txql_edge2",
                                "/data/BYD_dingzi/detection_data/history/01_sx_zp_nqp_txql_edge2_split")

