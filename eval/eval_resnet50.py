from models.backbone_resnet50 import Resnet50
import torch
from torchvision import transforms
import glob
import os
from PIL import Image


def eval_images(model_pth, image_path):
    test_transformer = transforms.Compose([
        transforms.Resize((256, 256), interpolation=3),
        # transforms.CenterCrop(size=opt.height),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    model = Resnet50(2)
    checkpoint = torch.load(model_pth, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint)
    model.eval()

    images = glob.glob(image_path)

    result = {"0": 0, "1": 0}
    for img_ in images:
        img_base_name = os.path.basename(img_)
        image_pure_name = os.path.splitext(img_base_name)[0]
        platform_id = image_pure_name.split("-")[-1]
        image = Image.open(img_)

        image = test_transformer(image)
        image = image.unsqueeze(0)

        outputs = model(image)
        _, preds = torch.max(outputs.data, 1)
        pred_class_id = int(preds)
        if pred_class_id == 1:
            result["1"] += 1
        else:
            result["0"] += 1

    print(result)


if __name__ == '__main__':
    eval_images("../_exports/resnet50_shangxian_0103.pth", "/home/log/Downloads/overkill_ddpm/fanguang_ddpm_500/*.jpg")