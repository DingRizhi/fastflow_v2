import yaml
from main import build_test_data_loader
import dataset
from models.fastflow_simply import FastFlowSimply
import torch
from postprocessing.caculate import predict_anomaly_score
from postprocessing.plot import generate_image


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


def eval_export_model(model_path, mode):
    test_dataset = dataset.MVTecDataset(
        root="/data/BYD_dingzi/dataset",
        category="duanziqiliu_crop_141",
        is_train=False,
        input_h=256,
        input_w=256
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )
    # dummy_input = torch.autograd.Variable(
    #     torch.randn(1, 3, 256, 256)
    # )
    jit_model = None
    if mode == "torchscript":
        jit_model = torch.jit.load(model_path)

        # output = jit_model.forward(dummy_input)
        # print(1)
    for n_iter, (data, labels) in enumerate(test_dataloader):
        inputs = data.cpu().detach()
        labels = labels.cpu().detach()
        outputs = jit_model.forward(inputs)
        outputs = outputs.cpu().detach()

        for n_batch, (anomaly_map, image, label) in enumerate(zip(outputs, inputs, labels)):
            score = predict_anomaly_score(anomaly_map)
            result_name = generate_image(anomaly_map, image, label, score,
                           n_batch, n_iter, "_exports", 0.776)
            print(result_name)
        break


if __name__ == '__main__':
    # convert_to_torchscript("configs/resnet18.yaml",
    #                 "_experiment_checkpoints/exp79_duanziqiliui_crop_141_2022-12-10-17-08/199.pt",
    #                 "_exports/fastflow_2.pth")

    eval_export_model("_exports/fastflow_2.pth", "torchscript")