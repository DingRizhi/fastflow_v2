import yaml
from main import build_test_data_loader
import dataset
from PIL import Image
from models.backbone_resnet50 import Resnet50
import torch
import onnxruntime
import os
import numpy as np
from dataloader.classify_dataset import ClassifyDataset
from torchvision import transforms
from postprocessing.caculate import predict_anomaly_score
from postprocessing.plot import generate_image
import glob
from scipy.special import softmax


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


def convert_to_resnet_onnx(model_pth, output_path, input_size=(256, 256)):

    model = Resnet50(2)
    checkpoint = torch.load(model_pth, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint)
    model.eval()
    dummy_input = torch.autograd.Variable(
        torch.randn(1, 3, input_size[0], input_size[1])
    )

    # with torch.no_grad():
    #     jit_model = torch.jit.trace(model, dummy_input)
    #     # jit_model = torch.jit.script(model, dummy_input)
    #     # print(jit_model.code)
    #     jit_model.save(output_path)

    # torch.onnx.export(
    #     model,
    #     dummy_input,
    #     output_path,
    #     verbose=True,
    #     keep_initializers_as_inputs=True,
    #     # opset_version=15,
    #     input_names=["data"],
    #     output_names=["output"],
    # )

    dynamic = True
    opset = 12

    torch.onnx.export(model, dummy_input, output_path, verbose=False,
                      opset_version=opset,
                      training=torch.onnx.TrainingMode.EVAL,
                      do_constant_folding=True,
                      input_names=['INPUT__0'],
                      output_names=['OUTPUT__0'],
                      dynamic_axes={'INPUT__0': {0: 'batch', 2: 'height', 3: 'width'},  # shape(1,3,640,640)
                                    'OUTPUT__0': {0: 'batch', 1: 'classes'}  # shape(1,25200,85)
                                    } if dynamic else None)


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


def inference(test_transformer, image, session):
    input_name = session.get_inputs()[0].name
    image = test_transformer(image)
    image = image.unsqueeze(0)
    image = to_numpy(image)

    outputs = session.run(None, {input_name: image})[0]
    outputs = softmax(outputs, axis=1)

    # print(outputs[0].shape, outputs[1].shape, outputs[2].shape)
    predict_ids = []
    for i in list(outputs):
        if i[0] > i[1]:
            predict_ids.append(0)
        else:
            predict_ids.append(1)
    print(outputs)
    return predict_ids


def check_export(onnx_path_demo, image_path, crop_flag=False):
    test_transformer = transforms.Compose([
        transforms.Resize((256, 256), interpolation=3),
        # transforms.CenterCrop(size=opt.height),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    session = onnxruntime.InferenceSession(onnx_path_demo)

    images = glob.glob(image_path)
    pred_result = []
    for img_ in images:
        img_base_name = os.path.basename(img_)
        image_pure_name = os.path.splitext(img_base_name)[0]
        s = image_pure_name.split("-")
        platform_id = s[-1]
        if '_' in platform_id:
            platform_id = platform_id.split("_")[0]
        image = Image.open(img_)
        print(img_)

        if crop_flag:
            crop_boxes = duanzi_crop_bbox[str(platform_id)]
            for index, crop_box in enumerate(crop_boxes):
                image_crop = image.crop((crop_box[0][0], crop_box[0][1], crop_box[1][0], crop_box[1][1]))
                # image_crop.save(os.path.join(os.path.dirname(img_), f"{image_pure_name}_crop_{index}.jpg"))
                predict_ids = inference(test_transformer, image_crop, session)
                pred_result += predict_ids

        else:
            predict_ids = inference(test_transformer, image, session)
            pred_result += predict_ids

            # print(outputs[0].shape, outputs[1].shape, outputs[2].shape)
    pre_1 = 0
    pre_0 = 0
    for i in pred_result:
        if i == 1:
            pre_1 += 1
        else:
            pre_0 += 1
    res = {"0": pre_0, "1": pre_1}
    print(res)


if __name__ == '__main__':
    # duanziqiliu
    convert_to_resnet_onnx("../_exports/resnet50_duanziqiliu_0111.pth",
                           "../_exports/resnet50_duanziqiliu_0111.onnx",
                           (256, 256))

    # shangxian
    # convert_to_resnet_onnx("../_exports/resnet50_shangxian_0106_2.pth",
    #                        "../_exports/resnet50_shangxian_0106_2.onnx",
    #                        (128, 128))

    # check_export("../_exports/resnet50_v2.onnx", "../_exports/0390-0024-94.jpg", True)
    # check_export("../_exports/resnet50_duanziqiliu_0107.onnx", "/data/BYD_dingzi/duanziqiliu/2023-01-05/good/*.jpg", False)
    # check_export("../_exports/resnet50_duanziqiliu_0110.onnx", "/data/BYD_dingzi/dataset/duanziqiliu_manual_crop_classify_v3/1_good/test/*.jpg", False)
    check_export("../_exports/resnet50_duanziqiliu_0111.onnx",
                 "/data/BYD_dingzi/duanziqiliu/2023-01-10/good/*.jpg", False)
