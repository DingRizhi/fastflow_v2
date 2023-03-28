import os
import glob
import json
import copy
import shutil


def filter_labels(labeled_img_dir, selected_labels):
    img_path_list = glob.glob(f"{labeled_img_dir}/*.jpg")

    for img_path in img_path_list:
        json_path = img_path.replace("jpg", "json")

        if not os.path.exists(json_path):
            print(f"not exists json path: {json_path}")
            continue

        with open(json_path, 'r') as f:
            label_info = json.load(f)
            new_label_info = copy.deepcopy(label_info)

            shapes = label_info["shapes"]

            for shape in shapes:
                label_name = shape["label"]
                if label_name in selected_labels:

            with open(save_json_path, 'w') as f2:
                json.dump(new_label_info, f2, indent=2)

            shutil.copyfile(img_path, os.path.join(save_dir, os.path.basename(img_path)))


if __name__ == '__main__':
    filter_labels("/data/BYD_dingzi/dataset/duanziqiliu/good", ["duanzi_good", "duanzi_bad"],
                  "/data/BYD_dingzi/dataset/duanziqiliu_bbox/good")