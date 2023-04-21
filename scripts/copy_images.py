import os
import glob
import random
import shutil
import tqdm


def get_ok_or_ng(img_path):
    try:
        img_dir_name = os.path.basename(os.path.dirname(img_path))
        return img_dir_name.split("_")[-1]
    except Exception:
        return None


def copy_images_by_json_file(source_img_path_list, json_dir):
    img_name_dict = {}
    for img_path in source_img_path_list:
        img_name_dict[f"{os.path.basename(img_path)}"] = img_path

    json_path_list = glob.glob(f"{json_dir}/*.json")
    for json_path in tqdm.tqdm(json_path_list):
        json_base_name = os.path.basename(json_path)
        img_name = json_base_name.replace(".json", ".jpg")

        img_path = img_name_dict[img_name]

        shutil.copyfile(img_path, os.path.join(json_dir, img_name))


def copy_images_by_jpg_file(source_json_path_list, img_dir):
    json_name_dict = {}
    for json_path in source_json_path_list:
        json_name_dict[f"{os.path.basename(json_path)}"] = json_path

    img_path_list = glob.glob(f"{img_dir}/*.jpg")
    for img_path in tqdm.tqdm(img_path_list):
        img_base_name = os.path.basename(img_path)
        json_name = img_base_name.replace(".jpg", ".json")

        json_path = json_name_dict[json_name]

        shutil.copyfile(json_path, os.path.join(img_dir, json_name))


def copy_image_path_list(image_path_list, save_dir, included_keys=None):
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    print(f"imgs: {len(image_path_list)}")
    not_exists_imgs = []

    for img_path in image_path_list:
        img_base_name = os.path.basename(img_path)
        json_base_name = img_base_name.replace("jpg", "json")
        img_type = get_ok_or_ng(img_path)

        if included_keys and len(included_keys) > 0:
            if img_type is None or img_type not in included_keys:
                continue

        if not os.path.exists(img_path):
            not_exists_imgs.append(img_path)
            print(f"{img_path} not exists")
            continue
        shutil.copyfile(img_path, os.path.join(save_dir, img_base_name))

        if os.path.exists(img_path.replace("jpg", "json")):
            shutil.copyfile(img_path.replace("jpg", "json"), os.path.join(save_dir, json_base_name))

    print(f"not exits imgs: {len(not_exists_imgs)}")


def sample_sub_dir_data(root_data, save_root, sample_rate):
    sub_dirs = os.listdir(root_data)
    for sd in sub_dirs:
        sub_dir_path = os.path.join(root_data, sd)
        img_path_list = glob.glob(f"{sub_dir_path}/*.jpg")
        img_path_list = random.sample(img_path_list, int(len(img_path_list) * sample_rate))

        save_dir_path = os.path.join(save_root, sd)
        copy_image_path_list(img_path_list, save_dir_path)


if __name__ == '__main__':
    # copy_image_path_list(glob.glob(f"/data/Data2Model/test_dataset/test_all/test_new/*/*.jpg"),
    #                      "/data/Data2Model/test")

    # copy_images_by_json_file(glob.glob(f"/data/Data2Model/data/train_val_new_zangwu/train/*/*.jpg"),
    #                          "/data/Data2Model/data/new_zangwu_test/train_high_score")

    copy_images_by_jpg_file(glob.glob(f"/data/Data2Model/data/train_val_new_zangwu/train/*/*.json"),
                            "/data/Data2Model/data/new_zangwu_test/train_high_score_gt")

    # sample_sub_dir_data("/data/Data2Model/train/train_ok/train_ok_2023-03-21_cropped",
    #                     "/data/Data2Model/train/train_ok/train_ok_2023-03-21_cropped_sampled",
    #                     0.3)


    # {
    #     "pengshang": 0.11,
    #     "bianxing": 0.06,
    #     "huashang": 0.1,
    #     "yise": 0.07,
    #     "cashang": 0.12,
    #     "gubao": 0.15,
    #     "molie": 0.12,
    #     "yashang": 0.08,
    #     "zhenyan": 0.12
    # }