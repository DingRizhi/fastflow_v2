det_loutong = [143, 144, 176, 177, 205, 206]
det_01_neiqiaopian_txql = [3, 6, 9, 12, 46, 49, 52, 55, 89, 92, 95, 98, 133, 136, 139, 142, 145, 152, 164, 167, 170, 173, 191, 194, 198, 201]
det_01_zhipo_shangxian = [2, 5, 8, 11, 24, 27, 28, 30, 31, 34, 35, 37, 38, 40, 43, 45, 48, 51, 54, 67, 70, 71, 73, 74, 77, 78, 80, 81, 83, 86, 88, 91, 94, 97, 111, 114, 115, 117, 118, 121, 122, 124, 125, 127, 130, 132, 135, 138, 141, 153, 154, 157, 158, 161, 163, 166, 169, 172, 181, 184, 185, 188, 190, 193, 197, 200, 208, 211, 212, 215]
det_01_shangxian_txql_edge = [1, 4, 7, 10, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 44, 47, 50, 53, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 87, 90, 93, 96, 99, 100, 101, 103, 104, 105, 106, 107, 108, 109, 110, 180]
det_02_wqp_txql = [25, 26, 29, 32, 33, 36, 39, 41, 42, 68, 69, 72, 75, 76, 79, 82, 84, 85, 112, 113, 116, 119, 120, 123, 126, 128, 129]
class_duanziqiliu = [102, 150, 151]
faqiabianxing = [24, 27, 28, 30, 31, 34, 35, 37, 38, 40, 43, 67, 70, 71, 73, 74, 77, 78, 80, 81, 83, 86, 114, 115, 117, 118, 121, 122, 124, 125, 127]
loutong_mian = [176]

all = det_loutong + det_01_neiqiaopian_txql + det_01_zhipo_shangxian + det_01_shangxian_txql_edge + det_02_wqp_txql + class_duanziqiliu + faqiabianxing + loutong_mian
all = sorted(list(set(all)))

print(all)

