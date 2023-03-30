import glob
import os


def delete_repeat_files(source_img_glob):
    imgs = glob.glob(source_img_glob)

    img_set = set()
    repeat_imgs = []

    for img_path in imgs:
        key = "_M_".join(img_path.rsplit("/", 2)[-2:])

        if key not in img_set:
            img_set.add(key)
        else:
            print(f"{img_path} is repeat")
            repeat_imgs.append(img_path)

    print(f"repeat num: {len(repeat_imgs)}")
    # for i in repeat_imgs:
    #     os.remove(i)


if __name__ == '__main__':
    delete_repeat_files("/data/Data2Model/train_split_new_v2_sample/*/*/*.jpg")