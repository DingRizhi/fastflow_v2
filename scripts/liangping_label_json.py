import json
import os
import glob


def replace_liangpin_labelme_label(json_path):
    with open(json_path, 'r') as f:
        label_info = json.load(f)

    label_info["shapes"] = []
    return label_info


def set_to_liangpin_labelme_label(json_path_list):
    print(f"total json: {len(json_path_list)}")

    for json_path in json_path_list:

        label_info = replace_liangpin_labelme_label(json_path)

        with open(json_path, 'w') as f:
            json.dump(label_info, f, indent=2)


if __name__ == '__main__':
    set_to_liangpin_labelme_label(
        glob.glob(f"/data/Data2Model/train/train_ok/train_ok_2023-04-07_guosha_cropped/*/*.json")
    )