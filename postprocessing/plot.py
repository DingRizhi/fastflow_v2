import os
import cv2
import math
import matplotlib.pyplot as plt
import numpy as np
import torch
from typing import Optional, Tuple

from postprocessing.caculate import predict_anomaly_score


def add_label(
    image: np.ndarray,
    label_name: str,
    color: Tuple[int, int, int],
    confidence: Optional[float] = None,
    font_scale: float = 5e-3,
    thickness_scale=1e-3,
):
    """Adds a label to an image.

    Args:
        image (np.ndarray): Input image.
        label_name (str): Name of the label that will be displayed on the image.
        color (Tuple[int, int, int]): RGB values for background color of label.
        confidence (Optional[float]): confidence score of the label.
        font_scale (float): scale of the font size relative to image size. Increase for bigger font.
        thickness_scale (float): scale of the font thickness. Increase for thicker font.

    Returns:
        np.ndarray: Image with label.
    """
    image = image.copy()
    img_height, img_width, _ = image.shape

    font = cv2.FONT_HERSHEY_PLAIN
    text = label_name if confidence is None else f"{label_name} ({confidence*100:.0f}%)"

    # get font sizing
    font_scale = min(img_width, img_height) * font_scale
    thickness = math.ceil(min(img_width, img_height) * thickness_scale)
    (width, height), baseline = cv2.getTextSize(text, font, fontScale=font_scale, thickness=thickness)

    # create label
    label_patch = np.zeros((height + baseline, width + baseline, 3), dtype=np.uint8)
    label_patch[:, :] = color
    cv2.putText(
        label_patch,
        text,
        (0, baseline // 2 + height),
        font,
        fontScale=font_scale,
        thickness=thickness,
        color=0,
        lineType=cv2.LINE_AA,
    )

    # add label to image
    image[: baseline + height, : baseline + width] = label_patch
    return image


def add_normal_label(image: np.ndarray, confidence: Optional[float] = None):
    """Adds the normal label to the image."""
    return add_label(image, "normal", (225, 252, 134), confidence)


def add_anomalous_label(image: np.ndarray, confidence: Optional[float] = None):
    """Adds the anomalous label to the image."""
    return add_label(image, "anomalous", (255, 100, 100), confidence)

def convert_image(image):
    image_add = image.squeeze()
    image_add = (image_add - image_add.min()) / np.ptp(image_add)
    image_add = image_add * 255
    image_add = image_add.numpy().astype(np.uint8)

    return image_add


def generate_image(anomaly_map, image, label, anomaly_score, n_batch, n_iter, save_dir, threshold, alpha=0.8, gamma=0):

    image_add = convert_image(image)
    image_add = np.moveaxis(image_add, 0, -1)

    anomaly_map_add = convert_image(anomaly_map)
    anomaly_map_add = cv2.applyColorMap(anomaly_map_add, cv2.COLORMAP_JET)
    anomaly_map_add = cv2.cvtColor(anomaly_map_add, cv2.COLOR_BGR2RGB)
    superimposed_map = cv2.addWeighted(anomaly_map_add, alpha, image_add, (1 - alpha), gamma)

    result_name = None

    if anomaly_score > threshold:  # Normal
        # superimposed_map = add_normal_label(superimposed_map, round(anomaly_score, 3))
        superimposed_map = add_normal_label(superimposed_map, round(anomaly_score, 3))
        result_name = "Normal"
    else:
        # superimposed_map = add_anomalous_label(superimposed_map, 1-round(anomaly_score, 3))
        superimposed_map = add_anomalous_label(superimposed_map, round(anomaly_score, 3))
        result_name = "Anomaly"

    if label == 1:
        check = "Normal"
    else:
        check = 'Abnormal'

    fig, ax = plt.subplot_mosaic([['origin', 'heatmap']], figsize=(7, 3.5))

    ax['origin'].imshow(image_add)
    ax['origin'].axis('off')
    ax['origin'].set_title('Original Image')

    ax['heatmap'].imshow(superimposed_map)
    ax['heatmap'].axis('off')
    ax['heatmap'].set_title('Heat Map Prediction')

    plt.close()
    os.makedirs(save_dir, exist_ok=True)
    fig.savefig(os.path.join(save_dir, 'img_' + str(n_iter) + '_' + str(n_batch) + '_' + check + '.png'))
    return result_name


def visualize_heatmap(model, dataloader, save_dir, threshold):
    normal_gt_num = 0
    normal_correct_num = 0
    anomaly_gt_num = 0
    anomaly_correct_num = 0
    total_num = 0
    for n_iter, (data, labels) in enumerate(dataloader):
        data = data.cuda()
        with torch.no_grad():
            ret = model(data)
        outputs = ret["anomaly_map"].cpu().detach()
        inputs = data.cpu().detach()
        labels = labels.cpu().detach()

        for n_batch, (anomaly_map, image, label) in enumerate(zip(outputs, inputs, labels)):
            score, label_ = predict_anomaly_score(anomaly_map)
            result_name = generate_image(anomaly_map, image, label, score,
                           n_batch, n_iter, save_dir, threshold)

            total_num += 1
            if int(label) == 1:
                normal_gt_num += 1
                if result_name == "Normal":
                    normal_correct_num += 1
            elif int(label) == 0:
                anomaly_gt_num += 1
                if result_name == "Anomaly":
                    anomaly_correct_num += 1
    print(f"evl result: \ntotal_num={total_num}, normal_accuracy={normal_correct_num}/{normal_gt_num}="
          f"{normal_correct_num/normal_gt_num}, "
          f"anomaly_accuracy={anomaly_correct_num}/{anomaly_gt_num}={anomaly_correct_num/anomaly_gt_num}")

