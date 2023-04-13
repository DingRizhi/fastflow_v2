import os
import glob
import random
import shutil


def copy_imgs(img_path_list, target_path):
    for img_path in img_path_list:
        img_base_name = os.path.basename(img_path)
        dir_name = os.path.basename(os.path.dirname(img_path))

        target_dir_path = os.path.join(target_path, dir_name)
        if not os.path.exists(target_dir_path):
            os.mkdir(target_dir_path)
        target_img_path = os.path.join(target_dir_path, img_base_name)

        if not os.path.exists(img_path.replace("jpg", "json")):
            print(f"no json file {img_path}")
            continue

        shutil.copyfile(img_path, target_img_path)
        shutil.copyfile(img_path.replace("jpg", "json"), target_img_path.replace("jpg", "json"))


def split_train_val(data_root, save_dir):

    sub_defects = os.listdir(data_root)

    for defect in sub_defects:
        defect_path = os.path.join(data_root, defect)

        images = glob.glob(f"{defect_path}/*/*.jpg")
        total_ = len(images)
        val_images = random.sample(images, int(total_ * 0.28))
        train_images = [i for i in images if i not in val_images]

        copy_imgs(val_images, os.path.join(save_dir, "val"))
        # copy_imgs(train_images, os.path.join(save_dir, "train"))


if __name__ == '__main__':
    split_train_val("/data/Data2Model/train/train_defect_guosha/guosha_0411_cropped",
                    "/data/Data2Model/train/train_defect_guosha/guosha_0411_cropped_sample")