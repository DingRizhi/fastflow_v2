import os
import glob
import json
import copy
import shutil
from PIL import Image
import tqdm


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


def split_label_images_by_img_path_list(img_path_list, save_dir, split_data_dir=False):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    for img_path in tqdm.tqdm(img_path_list):
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
            if shapes is not None and len(shapes) > 0:
                for index, shape in enumerate(shapes):
                    label_name = shape["label"]

                    # if label_name not in ["pengshang", "zhenyan"]:
                    #     continue
                    label_save_dir = os.path.join(save_dir, label_name)
                    if not os.path.exists(label_save_dir):
                        os.makedirs(label_save_dir, exist_ok=True)

                    if split_data_dir:
                        save_img_dir = os.path.join(label_save_dir, os.path.basename(os.path.dirname(img_path)))
                    else:
                        save_img_dir = label_save_dir

                    save_img_dir_base_name = os.path.basename(save_img_dir)
                    if "syn" in save_img_dir_base_name or "ok" in save_img_dir_base_name:
                        continue

                    if not os.path.exists(save_img_dir):
                        os.mkdir(save_img_dir)


                    save_image_path = os.path.join(save_img_dir, image_base_name)
                    # if os.path.exists(save_image_path) and os.path.exists(os.path.join(label_save_dir, image_base_name.replace("jpg", "json"))):
                    #     continue

                    shutil.copyfile(img_path, save_image_path)
                    shutil.copyfile(json_path, os.path.join(save_img_dir, image_base_name.replace("jpg", "json")))


def split_label_original_images(labeled_img_dir, save_dir):
    img_path_list = glob.glob(f"{labeled_img_dir}/*.jpg")
    split_label_images_by_img_path_list(img_path_list, save_dir)


def split_label_images_by_glob_path_list(img_path_list_array, save_dir):
    img_path_list = []
    for i in img_path_list_array:
        img_path_list += glob.glob(i)
    split_label_images_by_img_path_list(img_path_list, save_dir, True)


if __name__ == '__main__':
    # split_label_crop_images("/home/log/Downloads/0106-伤线过杀残胶数据-19pcs", "/data/BYD_dingzi/canjiao_crop_0106")
    # split_label_original_images("/data/Data2Model/test",
    #                             "/data/Data2Model/test_split")

    # split_label_original_images("/data/Data2Model/test", "")

    split_label_images_by_glob_path_list(
        ["/data/Data2Model/huawei_pc_2023-04-19_adt/infer_ins/test_diff_huashang_0.11/error/*.jpg"],
        "/data/Data2Model/huawei_pc_2023-04-19_adt/infer_ins/test_diff_huashang_0.11/error_split")

