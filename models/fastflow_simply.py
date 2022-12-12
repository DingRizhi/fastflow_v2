import FrEIA.framework as Ff
import FrEIA.modules as Fm
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

import constants as const


def subnet_conv_func(kernel_size, hidden_ratio):
    def subnet_conv(in_channels, out_channels):
        hidden_channels = int(in_channels * hidden_ratio)
        return nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size, padding="same"),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, out_channels, kernel_size, padding="same"),
        )

    return subnet_conv


def nf_fast_flow(input_chw, conv3x3_only, hidden_ratio, flow_steps, clamp=2.0):
    nodes = Ff.SequenceINN(*input_chw)
    for i in range(flow_steps):
        if i % 2 == 1 and not conv3x3_only:
            kernel_size = 1
        else:
            kernel_size = 3
        nodes.append(
            Fm.AllInOneBlock,
            subnet_constructor=subnet_conv_func(kernel_size, hidden_ratio),
            affine_clamping=clamp,
            permute_soft=False,
        )
    return nodes


class FastFlowSimply(nn.Module):
    def __init__(
        self,
        backbone_name,
        flow_steps,
        input_size,
        conv3x3_only=False,
        hidden_ratio=1.0,

    ):
        super(FastFlowSimply, self).__init__()
        assert (
            backbone_name in const.SUPPORTED_BACKBONES
        ), "backbone_name must be one of {}".format(const.SUPPORTED_BACKBONES)

        self.feature_extractor = timm.create_model(
            backbone_name,
            pretrained=True,
            features_only=True,
            out_indices=[1, 2, 3],
        )
        channels = self.feature_extractor.feature_info.channels()
        scales = self.feature_extractor.feature_info.reduction()

        # for transformers, use their pretrained norm w/o grad
        # for resnets, self.norms are trainable LayerNorm
        self.norms = nn.ModuleList()
        for in_channels, scale in zip(channels, scales):
            self.norms.append(
                nn.LayerNorm(
                    [in_channels, int(input_size[0] / scale), int(input_size[1] / scale)],
                    elementwise_affine=True,
                )
            )

        self.nf_flows = nn.ModuleList()
        for in_channels, scale in zip(channels, scales):
            self.nf_flows.append(
                nf_fast_flow(
                    [in_channels, int(input_size[0] / scale), int(input_size[1] / scale)],
                    conv3x3_only=conv3x3_only,
                    hidden_ratio=hidden_ratio,
                    flow_steps=flow_steps,
                )
            )
        self.input_size = input_size

    def forward(self, x):

        features = self.feature_extractor(x)
        features = [self.norms[i](feature) for i, feature in enumerate(features)]

        loss = 0
        outputs = []
        for i, feature in enumerate(features):
            output, log_jac_dets = self.nf_flows[i](feature)
            loss += torch.mean(
                0.5 * torch.sum(output**2, dim=(1, 2, 3)) - log_jac_dets
            )
            outputs.append(output)

        anomaly_map_list = []
        for output in outputs:
            log_prob = -torch.mean(output**2, dim=1, keepdim=True) * 0.5
            prob = torch.exp(log_prob)
            a_map = F.interpolate(
                -prob,
                size=[self.input_size[0], self.input_size[1]],
                mode="bilinear",
                align_corners=False,
            )
            anomaly_map_list.append(a_map)
        anomaly_map_list = torch.stack(anomaly_map_list, dim=-1)
        anomaly_map = torch.mean(anomaly_map_list, dim=-1)
        return anomaly_map
