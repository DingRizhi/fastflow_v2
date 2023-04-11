import os
import shutil
import glob
import tqdm


def merge_data_files(source_path, target_path):
    img_path_list = glob.glob(f"{source_path}/*/*.jpg")

    for img_path in tqdm.tqdm(img_path_list):
        img_base_name = os.path.basename(img_path)
        dir_name = os.path.basename(os.path.dirname(img_path))

        target_dir_path = os.path.join(target_path, dir_name)
        if not os.path.exists(target_dir_path):
            os.mkdir(target_dir_path)
        target_img_path = os.path.join(target_dir_path, img_base_name)

        if os.path.exists(target_img_path):
            print(f"replace {target_img_path}")
        else:
            print(f"### add a new {target_img_path}")
        shutil.copyfile(img_path, target_img_path)
        shutil.copyfile(img_path.replace("jpg", "json"), target_img_path.replace("jpg", "json"))


if __name__ == '__main__':
    merge_data_files("/data/Data2Model/huawei_pc_2023-04-04_clear/train_val_data_split/train/zhenyan",
                     "/data/Data2Model/huawei_pc_2023-04-04_clear/train_val_data_cropped/train")
