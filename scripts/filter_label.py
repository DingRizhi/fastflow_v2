import os
import glob
import json
import copy
import shutil
import random


def select_shapes_by_scores(label_info, min_score, max_score):
    shapes = label_info["shapes"]

    new_shapes = []
    for shape in shapes:
        label_name = shape["label"]
        score = shape["score"]

        if min_score <= score < max_score:
            new_shapes.append(shape)
    if len(new_shapes) == 0:
        return None

    sample_num = len(new_shapes) if len(new_shapes) < 4 else 4
    new_shapes = random.sample(new_shapes, sample_num)
    label_info["shapes"] = new_shapes

    return label_info


def filter_labels_by_score(source_json_dir, save_dir, min_score, max_score):
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    json_path_list = glob.glob(f"{source_json_dir}/*.json")

    for json_path in json_path_list:

        if not os.path.exists(json_path):
            print(f"not exists json path: {json_path}")
            continue

        with open(json_path, 'r') as f:
            label_info = json.load(f)
        label_info = select_shapes_by_scores(label_info, min_score, max_score)

        if label_info:
            new_json_path = os.path.join(save_dir, os.path.basename(json_path))
            with open(new_json_path, "w") as f:
                json.dump(label_info, f, indent=2)


if __name__ == '__main__':
    filter_labels_by_score("/data/Data2Model/train/train_ok_guosha/bak/D_guosha_json_0.15",
                           "/data/Data2Model/train/train_ok_guosha/bak/D_guosha_json_0.15_0.2_5",
                           0.15, 0.2)