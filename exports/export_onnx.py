import yaml
import os
from torchvision import transforms
from PIL import Image
from models.fastflow_simply import FastFlowSimply
import torch
from postprocessing.caculate import predict_anomaly_score
from postprocessing.plot import generate_image
from exports.export_resnet_onnx import to_numpy
import glob
import cv2

def convert_to_torchscript(config_path, model_pth, output_path):
    config = yaml.safe_load(open(config_path, "r"))
    # model = build_model(config)
    model = FastFlowSimply("resnet18", 8, [256, 256], True, 1.0)
    checkpoint = torch.load(model_pth, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    dummy_input = torch.autograd.Variable(
        torch.randn(1, 3, config["input_height"], config["input_width"])
    )

    with torch.no_grad():
        jit_model = torch.jit.trace(model, dummy_input)
        # jit_model = torch.jit.script(model, dummy_input)
        # print(jit_model.code)
        jit_model.save(output_path)

    # torch.onnx.export(
    #     model,
    #     dummy_input,
    #     output_path,
    #     verbose=True,
    #     keep_initializers_as_inputs=True,
    #     opset_version=15,
    #     input_names=["data"],
    #     output_names=["output"],
    # )


def eval_export_model(model_path, mode, image_path):
    image_transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    images = glob.glob(image_path)

    # dummy_input = torch.autograd.Variable(
    #     torch.randn(1, 3, 256, 256)
    # )
    jit_model = None
    if mode == "torchscript":
        jit_model = torch.jit.load(model_path)

    for img_ in images:
        print(os.path.basename(img_))
        image = Image.open(img_)
        # image = cv2.imread(img_)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # image = Image.fromarray(image)
        image = image_transform(image)
        image = image.unsqueeze(0)
        # print(image)

        outputs = jit_model.forward(image)
        outputs = outputs.cpu().detach()

        # print(f"outputs: {outputs}")

        score, label_ = predict_anomaly_score(outputs, threshold=0.388)
        # score1, label_1 = predict_anomaly_score(anomaly_map, region_box=[23, 118, 238, 213])

        # result_name = generate_image(anomaly_map, image, label, score,
        #                              n_batch, n_iter, "../_exports/faqiabianxing ", 0.388, region_box=1)
        print(f"label_name: {label_}, score: {score}")
        # break


if __name__ == '__main__':
    # convert_to_torchscript("../configs/resnet18.yaml",
    #                 "../_experiment_checkpoints/exp60_dingzi_side_data_2022-12-07-17-21/129.pt",
    #                 "../_exports/fastflow_faqiabianxing.pth")

    # eval_export_model("../_exports/fastflow_faqiabianxing.pth", "torchscript",
    #                   "/home/log/PycharmProjects/triton_deploy_cloud/triton_template/test_data/val_faqiabianxing_img/defect/0390-0022-23.jpg")

    eval_export_model("../_exports/fastflow_faqiabianxing.pth", "torchscript",
                      "/home/log/PycharmProjects/triton_deploy_cloud/triton_template/test_data/val_faqiabianxing_img/defect/*.jpg")

    # eval_export_model("../_exports/model.pt", "torchscript",
    #                   "/home/log/PycharmProjects/triton_deploy_cloud/triton_template/test_data/val_faqiabianxing_img/defect/0390-0022-23.jpg")