import glob
import os


def rename_images(data_dir, prefixed_name):
    image_path_list = glob.glob(f"{data_dir}/*.jpg")
    print(f"total images: {len(image_path_list)}")

    for image_path in image_path_list:
        dir_path = os.path.dirname(image_path)
        base_name = os.path.basename(image_path)
        new_img_name = f"{prefixed_name}_{base_name}"

        new_img_path = os.path.join(dir_path, new_img_name)

        os.rename(image_path, new_img_path)


if __name__ == '__main__':
    rename_images("/data/BYD_dingzi/dataset/loutong_168/test_2/good", "good")

