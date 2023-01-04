import yaml
import os
from torchvision import transforms
from PIL import Image
from models.fastflow_simply import FastFlowSimply
import torch
from postprocessing.caculate import predict_anomaly_score
from postprocessing.plot import generate_image
from postprocessing.plot import visualize_heatmap
from exports.export_resnet_onnx import to_numpy
import glob
import cv2
from main import build_model
import dataset


def eval_fastflow_model(config_path, model_path, threshold, save_dir):
    image_transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    input_size = [512, 512]
    test_dataset = dataset.MVTecDataset(
        root="/data/BYD_dingzi/dataset",
        category="loutong_168",
        # input_size=config["input_size"],
        is_train=False,
        input_h=input_size[0],
        input_w=input_size[1]
    )
    dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )

    config = yaml.safe_load(open(config_path, "r"))
    model = build_model(config)
    # model = FastFlowSimply("resnet18", 8, input_size, True, 1.0)
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.cuda()
    model.eval()

    # images = glob.glob(image_path)

    visualize_heatmap(model, dataloader, save_dir, threshold)


if __name__ == '__main__':
    eval_fastflow_model("../configs/resnet18.yaml", "../_experiment_checkpoints/exp115_loutong_168_2023-01-04-15-19/23.pt", 0.8198325037956238,
                        "../_eval/fastflow_loutong_168")