import yaml
from main import build_test_data_loader
import dataset
from models.backbone_resnet50 import Resnet50
import torch
from postprocessing.caculate import predict_anomaly_score
from postprocessing.plot import generate_image


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

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        verbose=True,
        keep_initializers_as_inputs=True,
        # opset_version=15,
        input_names=["data"],
        output_names=["output"],
    )


if __name__ == '__main__':
    convert_to_resnet_onnx("../_experiment_checkpoints/exp95_resnet50_2022-12-13-10-57/epoch_44_model.pth", "../_exports/resnet50.onnx")