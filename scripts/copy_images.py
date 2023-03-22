import os
import glob
import shutil


def get_ok_or_ng(img_path):
    try:
        img_dir_name = os.path.basename(os.path.dirname(img_path))
        return img_dir_name.split("_")[-1]
    except Exception:
        return None


def copy_image_path_list(image_path_list, save_dir, included_keys=None):
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    print(f"imgs: {len(image_path_list)}")

    for img_path in image_path_list:
        img_base_name = os.path.basename(img_path)
        json_base_name = img_base_name.replace("jpg", "json")
        img_type = get_ok_or_ng(img_path)

        if included_keys and len(included_keys) > 0:
            if img_type is None or img_type not in included_keys:
                continue

        shutil.copyfile(img_path, os.path.join(save_dir, img_base_name))

        if os.path.exists(img_path.replace("jpg", "json")):
            shutil.copyfile(img_path.replace("jpg", "json"), os.path.join(save_dir, json_base_name))


if __name__ == '__main__':
    copy_image_path_list(glob.glob(f"/data/Data2Model/test_original/*/*.jpg"), "/data/Data2Model/test")
