from PIL import Image
import glob
import os
import json


def crop_images(img_dir, save_dir):
    image_path_list = glob.glob(f"{img_dir}/*.jpg")

    for image_path in image_path_list:
        image_pure_name = os.path.splitext(os.path.basename(image_path))[0]

        image = Image.open(image_path)  # 读入当前图片
        img = image.convert('RGB')  # 转换成RGB三通道格式
        w = img.size[0]
        h = img.size[1]
        img_1 = img.crop([0, 0, w / 2, h / 2])
        img_1.save(os.path.join(save_dir, f"{image_pure_name}_0.jpg"))
        img_2 = img.crop([w / 2, 0, w, h / 2])
        img_2.save(os.path.join(save_dir, f"{image_pure_name}_1.jpg"))
        img_3 = img.crop([0, h / 2, w / 2, h])
        img_3.save(os.path.join(save_dir, f"{image_pure_name}_2.jpg"))
        img_4 = img.crop([w / 2, h / 2, w, h])
        img_4.save(os.path.join(save_dir, f"{image_pure_name}_3.jpg"))


def crop_images_in_labelme_images(img_dir, save_dir, good_dir_name="defect_good"):
    image_path_list = glob.glob(f"{img_dir}/*.jpg")
    good_dir = os.path.join(save_dir, good_dir_name)
    if not os.path.exists(good_dir):
        os.mkdir(good_dir)
    bad_dir = os.path.join(save_dir, "defect_bad")
    if not os.path.exists(bad_dir):
        os.mkdir(bad_dir)

    for image_path in image_path_list:
        print(image_path)
        image_pure_name = os.path.splitext(os.path.basename(image_path))[0]

        label_path = image_path.replace("jpg", "json")
        image = Image.open(image_path)
        with open(label_path, "r") as f:
            label_infos = json.load(f)
            shapes = label_infos["shapes"]

            for index, shape in enumerate(shapes):
                points = shape["points"]
                label = shape["label"]
                x1, y1, x2, y2 = 65535, 65535, -65535, -65535
                for point in points:
                    x1, y1, x2, y2 = (
                        min(x1, point[0]),
                        min(y1, point[1]),
                        max(x2, point[0]),
                        max(y2, point[1])
                    )
                print(x1, y1, x2, y2)
                img_ = image.crop((x1, y1, x2, y2))
                if label == "good":
                    img_.save(os.path.join(good_dir, f"{image_pure_name}_{index}.jpg"))
                elif label == "bad":
                    img_.save(os.path.join(bad_dir, f"{image_pure_name}_{index}.jpg"))


if __name__ == '__main__':
    crop_images("/home/log/PycharmProjects/fastflow_v2/datasets/MVTec/loutong_base/train/good",
                "/home/log/PycharmProjects/fastflow_v2/datasets/MVTec/luotong_4_split/train/good")

    # crop_images_in_labelme_images("/home/log/PycharmProjects/fastflow_v2/datasets/MVTec/loutong_test_original/defect",
    #                               "/home/log/PycharmProjects/fastflow_v2/datasets/MVTec/loutong_test_original/")
