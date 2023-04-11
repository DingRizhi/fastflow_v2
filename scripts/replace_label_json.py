import os
import json
import glob
import shutil


def replace_labelme_label(json_path, replace_dict):
    with open(json_path, 'r') as f:
        label_info = json.load(f)

    shapes = label_info["shapes"]
    new_shapes = []
    for index, shape in enumerate(shapes):
        label_name = shape["label"]

        if label_name in replace_dict.keys():
            shape["label"] = replace_dict[label_name]
        new_shapes.append(shape)
    label_info["shapes"] = new_shapes
    return label_info


def replace_labelme_json_with_img_paths(img_path_list, save_dir, replace_dict, copy_img=True):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    for img_path in img_path_list:
        json_path = img_path.replace("jpg", "json")
        image_base_name = os.path.basename(img_path)
        image_pure_name = os.path.splitext(image_base_name)[0]

        label_info = replace_labelme_label(json_path, replace_dict)

        new_json_path = os.path.join(save_dir, image_base_name.replace("jpg", "json"))
        new_img_path = os.path.join(save_dir, image_base_name)
        with open(new_json_path, 'w') as f:
            json.dump(label_info, f, indent=2)

        if copy_img and img_path != new_img_path:
            shutil.copyfile(img_path, new_img_path)


def change_labelme_label_to_target_dir(source_path_glob, save_dir, replace_dict):
    img_path_list = glob.glob(source_path_glob)
    print(f"total image: {len(img_path_list)}")

    replace_labelme_json_with_img_paths(img_path_list, save_dir, replace_dict)


def modify_labelme_label(source_path_glob, replace_dict):
    img_path_list = glob.glob(source_path_glob)
    print(f"total image: {len(img_path_list)}")

    for img_path in img_path_list:
        json_path = img_path.replace("jpg", "json")
        image_base_name = os.path.basename(img_path)
        # image_pure_name = os.path.splitext(image_base_name)[0]

        label_info = replace_labelme_label(json_path, replace_dict)

        with open(json_path, 'w') as f:
            json.dump(label_info, f, indent=2)


if __name__ == '__main__':
    modify_labelme_label("/data/Data2Model/huawei_pc_2023-04-04_clear_v2/train_val_data_cropped/*/*/*.jpg",
                         {"yise1": "yise", "yise2": "yise", "yashnag": "yashang"})