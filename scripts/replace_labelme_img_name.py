import json
import os
import glob


def replace_labelme_img_name(json_path):
    image_name = os.path.basename(json_path).replace(".json", ".jpg")
    with open(json_path, 'r') as f:
        label_info = json.load(f)

    need_write = False
    if image_name != label_info["imagePath"]:
        print(f"replace img name: {label_info['imagePath']} --> {image_name}")
        label_info["imagePath"] = image_name
        need_write = True
    return label_info, need_write


def replace_labelme_img_name_with_img_paths(img_path_list):
    count = 0
    for img_path in img_path_list:
        json_path = img_path.replace("jpg", "json")
        image_base_name = os.path.basename(img_path)

        label_info, need_write = replace_labelme_img_name(json_path)

        if need_write:
            count += 1
            print(f"rewrite: {json_path}")
            with open(json_path, 'w') as f:
                json.dump(label_info, f, indent=2)
    print(f"count = {count}")


def modify_all_img_name(glob_path):
    img_path_list = glob.glob(glob_path)
    print(f"total image: {len(img_path_list)}")

    replace_labelme_img_name_with_img_paths(img_path_list)


if __name__ == '__main__':
    modify_all_img_name("/data/Data2Model/train_split_new_v2/yise/*/*.jpg")