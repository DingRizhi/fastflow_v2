import os

from openpyxl import load_workbook
from excel_tool import ParseExcel
from copy_images import copy_image_path_list
import shutil


HUAWEI_PC_TEST = "/data/Data2Model/test"


def build_image_name(row):
    b_task_id = row[1].value
    c_product_id = row[2].value
    d_image_id = row[3].value

    image_name = f"{str(b_task_id).rjust(4, '0')}-{str(c_product_id).rjust(4, '0')}-{str(d_image_id).rjust(2, '0')}.jpg"
    # print(image_name)
    return image_name


def parse_huawei_pc(sheet):
    save_dir = "/data/Data2Model/test_analyze/v5_loujian"
    loujian_img_path_list = []
    for row_index, row in enumerate(sheet.iter_rows()):
        if row_index == 0:
            continue

        o_result_type = row[14]

        if o_result_type.value == "模型漏检":
            image_name = build_image_name(row)
            image_path = os.path.join(HUAWEI_PC_TEST, image_name)
            loujian_img_path_list.append(image_path)

    print(len(loujian_img_path_list))
    copy_image_path_list(loujian_img_path_list, save_dir)


if __name__ == '__main__':
    p = ParseExcel("/home/logding/Downloads/模型报告-HW_PC-D-测试集-V3-V4-V5.xlsx")
    sh = p.wb["v5"]
    parse_huawei_pc(sh)
