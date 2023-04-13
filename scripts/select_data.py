import os
import tqdm
import json
import shutil
import glob


def select_only_one_defect_data(img_path_list, save_dir, defect_name):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    for img_path in tqdm.tqdm(img_path_list):
        json_path = img_path.replace("jpg", "json")
        # image = Image.open(img_path)
        image_base_name = os.path.basename(img_path)
        image_pure_name = os.path.splitext(image_base_name)[0]
        if not os.path.exists(json_path):
            print(f"not exists json path: {json_path}")
            continue

        with open(json_path, 'r') as f:
            label_info = json.load(f)

            shapes = label_info["shapes"]
            if shapes is not None and len(shapes) > 0:
                flag = True
                label_name = None
                for index, shape in enumerate(shapes):
                    label_name = shape["label"]
                    if label_name != defect_name:
                        flag = False
                        break

                if flag and label_name is not None:
                    label_save_dir = os.path.join(save_dir, label_name)
                    save_img_dir = os.path.join(label_save_dir, os.path.basename(os.path.dirname(img_path)))
                    if not os.path.exists(save_img_dir):
                        os.makedirs(save_img_dir, exist_ok=True)
                    save_image_path = os.path.join(save_img_dir, image_base_name)
                    shutil.copyfile(img_path, save_image_path)
                    shutil.copyfile(json_path, os.path.join(save_img_dir, image_base_name.replace("jpg", "json")))


if __name__ == '__main__':
    select_only_one_defect_data(glob.glob("/data/Data2Model/data/train_new_zangwu/train_new/*/*.jpg"),
                                "/data/Data2Model/data/train_new_zangwu_split", "zangwu")