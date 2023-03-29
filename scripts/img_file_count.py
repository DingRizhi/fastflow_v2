import os
import glob
import random
import shutil


def count_sub_dir_files(root_dir):
    sub_defects = os.listdir(root_dir)

    count_dict = {}

    for defect in sub_defects:
        defect_dir_path = os.path.join(root_dir, defect)

        imgs = glob.glob(f"{defect_dir_path}/*/*.jpg")

        count_dict[defect] = len(imgs)

    for k, v in count_dict.items():
        print(f"{k}: {v}")


def sample_data_img_files(root_dir, save_dir, defect_name, sample_num):
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    imgs = glob.glob(f"{root_dir}/{defect_name}/*/*.jpg")
    sample_imgs = random.sample(imgs, sample_num)

    save_defect_dir = os.path.join(save_dir, defect_name)
    if not os.path.exists(save_defect_dir):
        os.mkdir(save_defect_dir)

    for img_path in sample_imgs:
        image_base_name = os.path.basename(img_path)
        save_img_dir = os.path.join(save_defect_dir, os.path.basename(os.path.dirname(img_path)))
        if not os.path.exists(save_img_dir):
            os.mkdir(save_img_dir)

        save_image_path = os.path.join(save_img_dir, image_base_name)
        shutil.copyfile(img_path, save_image_path)
        shutil.copyfile(img_path.replace("jpg", "json"),
                        os.path.join(save_img_dir, image_base_name.replace("jpg", "json")))


if __name__ == '__main__':
    # count_sub_dir_files("/data/Data2Model/train_split_new_v2")

    sample_data_img_files("/data/Data2Model/train_split_new_v2",
                          "/data/Data2Model/train_split_new_v2",
                          "pengshang", 700)

