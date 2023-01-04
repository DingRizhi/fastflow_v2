import os
import glob
import shutil
import random


def extract_images_by_platform_ids(image_dir, save_dir, platform_ids, merge_save=False, flag_str=None):
    image_path_list = glob.glob(f"{image_dir}/*.jpg")
    if flag_str is not None:
        save_dir = os.path.join(save_dir, flag_str)
        os.makedirs(save_dir, exist_ok=True)

    for image_path in image_path_list:
        image_base_name = os.path.basename(image_path)
        image_pure_name, image_exe_name = os.path.splitext(image_base_name)
        image_platform_id = int(image_pure_name.split("-")[-1])
        if image_platform_id in platform_ids:
            if merge_save:
                shutil.copyfile(image_path, os.path.join(save_dir, image_base_name))
                print(f"copy image to {os.path.join(save_dir, image_base_name)}")
            else:
                save_platform_id_dir = os.path.join(save_dir, str(image_platform_id))
                os.makedirs(save_platform_id_dir, exist_ok=True)
                shutil.copyfile(image_path, os.path.join(save_platform_id_dir, image_base_name))
                print(f"copy image to {os.path.join(save_platform_id_dir, image_base_name)}")


def copy_images(img_path_list, mode, save_dir):
    for image_path in img_path_list:
        image_base_name = os.path.basename(image_path)
        img_good_dir = os.path.join(save_dir, mode, "good")

        os.makedirs(img_good_dir, exist_ok=True)

        shutil.copyfile(image_path, os.path.join(img_good_dir, image_base_name))


def split_data(original_img_dir, data_dir, test_good_num=4):
    image_path_list = glob.glob(f"{original_img_dir}/*.jpg")

    platform_id_count_dict = {}
    for image_path in image_path_list:
        image_base_name = os.path.basename(image_path)
        image_pure_name, image_exe_name = os.path.splitext(image_base_name)
        image_platform_id = int(image_pure_name.split("-")[-1])

        if image_platform_id not in platform_id_count_dict:
            platform_id_count_dict[image_platform_id] = []
        else:
            platform_id_count_dict[image_platform_id].append(image_path)

    platform_id_image_split = {}
    for k, v in platform_id_count_dict.items():
        test_good_list = random.sample(v, test_good_num)
        train_good_list = [i for i in v if i not in test_good_list]

        platform_id_image_split[k] = {'train': train_good_list, 'test': test_good_list}

    for img_platform_id, img_split_dict in platform_id_image_split.items():
        test_good_list = img_split_dict['test']
        copy_images(test_good_list, 'test', data_dir)
        train_good_list = img_split_dict['train']
        copy_images(train_good_list, 'train', data_dir)


if __name__ == '__main__':
    # extract_images_by_platform_ids("/data/BYD_dingzi/12个产品良品图", "/data/BYD_dingzi/dataset/duanziqiliu/good",
    #                                [94, 140, 141])

    extract_images_by_platform_ids("/data/BYD_dingzi/achive/20221219/original_pictures-24pcs-良品", "/data/BYD_dingzi/loutong_big/good_original",
                                   [164, 167, 168], flag_str='20221219_24pcs_2')

    # split_data("/data/BYD_dingzi/dataset/duanziqiliu/good", "/data/BYD_dingzi/dataset/duanziqiliu")