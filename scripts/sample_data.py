import glob
import os
import shutil
import random


def sample_images(data_dir, save_dir, sample_num=150):
    img_path_list = glob.glob(f"{data_dir}/*.jpg")

    sample_image_path = random.sample(img_path_list, sample_num)

    for image_path in sample_image_path:
        image_base_name = os.path.basename(image_path)
        shutil.copyfile(image_path, os.path.join(save_dir, image_base_name))


if __name__ == '__main__':
    sample_images("/data/BYD_dingzi/fanguang_crop_0103/fanguang", "/data/BYD_dingzi/fanguang_crop_0103/sample_150")