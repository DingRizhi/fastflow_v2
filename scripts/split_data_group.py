import os
import glob
import shutil


def load_platform_ids(txt_path):
    f = open(txt_path, 'r')
    platform_ids = f.readlines()
    platform_ids = [i.strip() for i in platform_ids]
    print(sorted([int(i) for i in platform_ids]))
    f.close()
    return platform_ids


def split_byd_yolo_val_data(val_img_dir, txt_path, root_dir):
    platform_ids = load_platform_ids(txt_path)

    img_path_list = glob.glob(f"{val_img_dir}/*/*.jpg")

    group_save_dir = os.path.join(root_dir, os.path.basename(txt_path).split(".")[0])
    os.makedirs(group_save_dir, exist_ok=True)
    for img_path in img_path_list:
        img_base_name = os.path.basename(img_path)
        img_pure_name = os.path.splitext(img_base_name)[0]
        img_platform_id = img_pure_name.split("-")[-1]

        json_file_path = img_path.replace("jpg", "json")
        if img_platform_id in platform_ids:
            save_image_path = os.path.join(group_save_dir, img_base_name)

            shutil.copyfile(img_path, save_image_path)
            if os.path.exists(json_file_path):
                save_json_path = os.path.join(group_save_dir, img_base_name.replace("jpg", "json"))
                shutil.copyfile(json_file_path, save_json_path)


if __name__ == '__main__':
    # split_byd_yolo_val_data("/home/log/PycharmProjects/BYD_data/val",
    #                         "/home/log/PycharmProjects/BYD_data/03_sx_zp_txql.txt",
    #                         "/home/log/PycharmProjects/BYD_data")

    load_platform_ids("/home/log/PycharmProjects/BYD_data/02_wqp_txql.txt")