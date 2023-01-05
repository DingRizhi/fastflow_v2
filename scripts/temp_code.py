# coco_class_labels = ('background',
#                         'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
#                         'boat', 'traffic light', 'fire hydrant', 'street sign', 'stop sign',
#                         'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
#                         'elephant', 'bear', 'zebra', 'giraffe', 'hat', 'backpack', 'umbrella',
#                         'shoe', 'eye glasses', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
#                         'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
#                         'skateboard', 'surfboard', 'tennis racket', 'bottle', 'plate', 'wine glass',
#                         'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
#                         'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
#                         'couch', 'potted plant', 'bed', 'mirror', 'dining table', 'window', 'desk',
#                         'toilet', 'door', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
#                         'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'blender', 'book',
#                         'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush')
#
# coco_class_index = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20,
#                     21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44,
#                     46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 67,
#                     70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90]
#
# print(len(coco_class_index))
# print(len(coco_class_labels))

det_01 = [1, 4, 7, 10, 13, 15, 16, 17, 18, 19, 40, 43, 46, 49, 52, 53, 54, 55, 56, 58, 59, 80, 83, 86, 89, 92, 93, 95,
          96, 97, 98, 99, 3, 6, 9, 12, 42, 45, 48, 51, 82, 85, 88, 91, 122, 125, 128, 131, 154, 157, 160, 163, 184, 187,
          190, 193]
det_01_b = [2, 5, 8, 11, 14, 20, 23, 24, 26, 27, 30, 31, 33, 34, 36, 39, 41, 44, 47, 50, 60, 63, 64, 66, 67, 70, 71, 73,
            74, 76, 79, 81, 84, 87, 90, 100, 103, 104, 106, 107, 110, 111, 113, 114, 116, 119, 121, 124, 127, 130, 143,
            144, 147, 148, 151, 153, 156, 159, 162, 174, 175, 176, 177, 178, 179, 180, 181, 183, 186, 189, 192, 204,
            205, 206, 207, 208, 209, 210, 211]

print(tuple(det_01))
