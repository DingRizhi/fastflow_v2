from PIL import Image
import glob
import os
import json

duanzi_crop_bbox = {
    "102": [[[120, 1020], [850, 1903]],
           [[818, 1050], [1550, 1913]],
           [[1450, 1100], [2250, 1920]]],
    "150": [[[331, 1174], [1000, 2012]],
            [[927, 1125], [1536, 1994]],
            [[1529, 1099], [2209, 1905]]],
    "151": [[[222, 819], [1000, 1778]],
            [[870, 769], [1644, 1700]],
            [[1580, 688], [2338, 1600]]]
}


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


def crop_images_in_labelme_images(img_dir, save_dir, label_name_dict={"good":"duanzi_good", "bad":"duanzi_bad"}):
    image_path_list = glob.glob(f"{img_dir}/*.jpg")
    # good_dir = os.path.join(save_dir, "good")
    # if not os.path.exists(good_dir):
    #     os.mkdir(good_dir)
    # bad_dir = os.path.join(save_dir, "defect")
    # if not os.path.exists(bad_dir):
    #     os.mkdir(bad_dir)

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
                # if label == label_name_dict["good"]:
                #     img_.save(os.path.join(good_dir, f"{image_pure_name}_{index}.jpg"))
                # elif label == label_name_dict["bad"]:
                #     img_.save(os.path.join(bad_dir, f"{image_pure_name}_{index}.jpg"))
                if label == "shangxian":
                    img_.save(os.path.join(save_dir, f"{image_pure_name}_{index}.jpg"))



def crop_images_in_specific_bbox(img_dir, save_dir, crop_boxes_dict=duanzi_crop_bbox):
    image_path_list = glob.glob(f"{img_dir}/*.jpg")

    for image_path in image_path_list:
        print(image_path)
        image_pure_name = os.path.splitext(os.path.basename(image_path))[0]
        platform_id = image_pure_name.split("-")[-1]
        image = Image.open(image_path)
        crop_boxes = crop_boxes_dict[str(platform_id)]
        for i, points in enumerate(crop_boxes):
            img_ = image.crop((points[0][0], points[0][1], points[1][0], points[1][1]))
            img_.save(os.path.join(save_dir, f"{image_pure_name}_{i}.jpg"))



if __name__ == '__main__':
    # crop_images_in_labelme_images("/home/log/PycharmProjects/fastflow_v2/datasets/MVTec/dingzi_side_data/test/defect",
    #                               "/home/log/PycharmProjects/fastflow_v2/datasets/MVTec/dingzi_side_data_crop/test/defect")

    # crop_images_in_labelme_images("/home/log/PycharmProjects/fastflow_v2/datasets/MVTec/loutong_test_original/defect",
    #                               "/home/log/PycharmProjects/fastflow_v2/datasets/MVTec/loutong_test_original/")

    # crop_images_in_labelme_images("/home/log/Downloads/LIN_TEST/val_01","/data/BYD_dingzi/shangxian_crop/val_01")

    # crop_images_in_labelme_images("/data/BYD_dingzi/dataset/duanziqiliu/test/defect",
    #                               "/data/BYD_dingzi/dataset/duanziqiliui_crop_v2/")

    crop_images_in_specific_bbox("/data/BYD_dingzi/duanziqiliu/2023-01-10/original", "/data/BYD_dingzi/duanziqiliu/2023-01-10/crop")