import FrEIA.framework as Ff
import FrEIA.modules as Fm
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

import constants as const


class Resnet18(nn.Module):
    def __init__(self, class_num=2):
        super(Resnet18, self).__init__()
        self.class_num = class_num

        self.feature_extractor = timm.create_model(
            "resnet18",
            pretrained=True,
            features_only=True,
            out_indices=[3],
        )
        self.classifier = nn.Sequential(
            nn.Conv2d(256, class_num, kernel_size=1, stride=1, padding=0),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

    def forward(self, x):
        features = self.feature_extractor(x)
        out = self.classifier(features[0])
        out = out.view(-1, self.class_num)
        return out


# ----------------------model test--------------------------
def eval_model():
    a = torch.rand(8, 3, 256, 256)
    model = Resnet18(2)
    out = model(a)
    print(1)


if __name__ == '__main__':

    eval_model()