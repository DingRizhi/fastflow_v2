import glob
import os
import shutil
import random
from tqdm import tqdm


def sample_images(data_dir, save_dir, sample_num=150, copy_label=True):
    img_path_list = glob.glob(f"{data_dir}/*.jpg")

    sample_image_path = random.sample(img_path_list, sample_num)

    for image_path in tqdm(sample_image_path):
        image_base_name = os.path.basename(image_path)
        shutil.copyfile(image_path, os.path.join(save_dir, image_base_name))
        json_path = image_path.replace(".jpg", ".json")
        if copy_label and os.path.exists(json_path):
            shutil.copyfile(json_path, os.path.join(save_dir, os.path.basename(json_path)))


if __name__ == '__main__':
    sample_images("/data/ikku_mtl_data/test_o_0106", "/data/ikku_mtl_data/micro_o_test", sample_num=200)