import os


def get_class_names_by_file(class_name_file):
    with open(class_name_file, "r") as f:
        class_names = f.readlines()
        class_names = [i.strip() for i in class_names]
        class_names = sorted(class_names)
    return class_names


def diff_class_names(old_class_file, new_class_file):
    old_class_names = get_class_names_by_file(old_class_file)
    new_class_names = get_class_names_by_file(new_class_file)

    miss_class_names = [i for i in old_class_names if i not in new_class_names]
    new_add_class_names = [i for i in new_class_names if i not in old_class_names]

    print(f"缺失{len(miss_class_names)}个缺陷:\n{miss_class_names}")
    print(f"新增{len(new_add_class_names)}个缺陷:\n{new_add_class_names}")


if __name__ == '__main__':
    diff_class_names("/data/Data2Model/huawei_pc_2023-04-12_zangwu/class_names_list.txt",
                     "/data/Data2Model/huawei_pc_2023-04-21_defect/class_names_list.txt")