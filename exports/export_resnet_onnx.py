import yaml
from main import build_test_data_loader
import dataset
from PIL import Image
from models.backbone_resnet50 import Resnet50
import torch
import onnxruntime
from dataloader.classify_dataset import ClassifyDataset
from torchvision import transforms
from postprocessing.caculate import predict_anomaly_score
from postprocessing.plot import generate_image
import glob
from scipy.special import softmax


def convert_to_resnet_onnx(model_pth, output_path):

    model = Resnet50(2)
    checkpoint = torch.load(model_pth, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint)
    model.eval()
    dummy_input = torch.autograd.Variable(
        torch.randn(1, 3, 256, 256)
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


def check_export(onnx_path_demo, image_path):
    test_transformer = transforms.Compose([
        transforms.Resize((256, 256), interpolation=3),
        # transforms.CenterCrop(size=opt.height),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    session = onnxruntime.InferenceSession(onnx_path_demo)

    images = glob.glob(image_path)
    for img_ in images:
        image = Image.open(img_)
        image = test_transformer(image)
        image = image.unsqueeze(0)
        image = to_numpy(image)

        input_name = session.get_inputs()[0].name
        outputs = session.run(None, {input_name: image})[0]
        outputs = softmax(outputs, axis=1)

        # print(outputs[0].shape, outputs[1].shape, outputs[2].shape)
        print(img_)
        print(outputs)


if __name__ == '__main__':
    # convert_to_resnet_onnx("../_experiment_checkpoints/exp102_resnet50_2022-12-19-15-49/epoch_80_model.pth", "../_exports/resnet50_v2.onnx")

    check_export("../_exports/resnet50_v2.onnx", "../_exports/*.jpg")