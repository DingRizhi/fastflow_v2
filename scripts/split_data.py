import os
import glob
import random
import shutil


def copy_images(image_list, mode, save_dir, copy_json=False):
    save_dir = os.path.join(save_dir, mode)
    os.makedirs(save_dir, exist_ok=True)
    for image_path in image_list:
        image_base_name = os.path.basename(image_path)

        shutil.copyfile(image_path, os.path.join(save_dir, image_base_name))
        if copy_json:
            json_path = image_path.replace("jpg", "json")
            shutil.copyfile(json_path, os.path.join(save_dir, image_base_name.replace("jpg", "json")))


def split_coco_data(data_dir, save_root, train_num, val_num):
    image_path_list = glob.glob(f"{data_dir}/*.jpg")
    print(f"total images: {len(image_path_list)}")

    train_list = random.sample(image_path_list, train_num)
    remaining_list = [i for i in image_path_list if i not in train_list]
    val_list = random.sample(remaining_list, val_num)
    copy_images(train_list, "train2017", save_root)
    copy_images(val_list, "val2017", save_root)


def split_anomaly_data(data_dir, save_root, val_num):
    image_path_list = glob.glob(f"{data_dir}/*.jpg")
    print(f"total images: {len(image_path_list)}")

    val_num = int(0.2 * len(image_path_list)) if val_num is None else val_num
    val_list = random.sample(image_path_list, val_num)
    train_list = [i for i in image_path_list if i not in val_list]
    copy_images(train_list, "train", save_root)
    copy_images(val_list, "test", save_root)


if __name__ == '__main__':
    # split_anomaly_data("/data/BYD_dingzi/dataset/duanziqiliui_crop_v2/good",
    #                    "/data/BYD_dingzi/dataset/duanziqiliui_crop_v2", 30)

    split_anomaly_data("/home/log/Downloads/overkill_ddpm/fanguang_ddpm_500",
                       "/home/log/Downloads/overkill_ddpm/fanguang", None)
