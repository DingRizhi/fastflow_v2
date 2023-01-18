import os
import json
import glob
import shutil


def replace_labelme_json_label(label_dir, save_dir, replace_dict):
    img_path_list = glob.glob(f"{label_dir}/*.jpg")
    print(f"total image: {len(img_path_list)}")

    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    for img_path in img_path_list:
        json_path = img_path.replace("jpg", "json")
        image_base_name = os.path.basename(img_path)
        image_pure_name = os.path.splitext(image_base_name)[0]

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

        new_json_path = os.path.join(save_dir, image_base_name.replace("jpg", "json"))
        new_img_path = os.path.join(save_dir, image_base_name)
        with open(new_json_path, 'w') as f:
            json.dump(label_info, f, indent=2)

        shutil.copyfile(img_path, new_img_path)


if __name__ == '__main__':
    replace_labelme_json_label("/data/BYD_dingzi/detection_data/history/01_sx_zp_nqp_txql_edge4",
                               "/data/BYD_dingzi/detection_data/01_sx_zp_nqp_txql_edge2",
                               {"tiexinqiliu_edge3": "tiexinqiliu_edge1", "tiexinqiliu_edge4": "tiexinqiliu_edge2"})