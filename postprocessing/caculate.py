import numpy as np
from ignite.contrib import metrics
from sklearn.metrics import roc_auc_score
import torch

def predict_anomaly_score(anomaly_map):

    anomaly_map_x = anomaly_map.squeeze()
    anomaly_map_x = (anomaly_map_x - anomaly_map_x.min()) / np.ptp(anomaly_map_x)
    # print(anomaly_map_x,'+++')
    anomaly_score = anomaly_map_x.reshape(anomaly_map_x.shape[0], -1).max(dim=1).values[0].item()

    return anomaly_score


auroc_metric = metrics.ROC_AUC()
rpc_metric = metrics.PrecisionRecallCurve()


def get_best_thredhold(model, dataloader):

    for n_iter, (data, labels) in enumerate(dataloader):
        data = data.cuda()
        with torch.no_grad():
            ret = model(data)
            # print(ret)
        outputs = ret["anomaly_map"].cpu().detach()
        inputs = data.cpu().detach()
        labels = labels.cpu().detach()
        score_list = []
        label_list = []

        for n_batch, (anomaly_map, image, label) in enumerate(zip(outputs, inputs, labels)):
            score = predict_anomaly_score(anomaly_map)
            score_list.append(score)
            label_list.append(label.tolist())

        score_tensor = torch.Tensor(score_list)
        label_tensor = torch.Tensor(label_list)

        auroc_metric.update((score_tensor, label_tensor.to(torch.int)))
        rpc_metric.update((score_tensor, label_tensor.to(torch.int)))

    auroc = auroc_metric.compute()
    precision, recall, thresholds = rpc_metric.compute()
    beta = 1
    f_score = (1 + beta ** 2) * (precision * recall) / \
              (beta ** 2 * precision + recall + 1e-10)

    if thresholds.dim() == 0:
        threshold_best = thresholds
    else:
        threshold_best = thresholds[torch.argmax(f_score)]
    f_score_max = f_score[torch.argmax(f_score)]
    print("AUROC: {}; F-SCORE: {}; THRESHOLD: {}".format(auroc, f_score_max, threshold_best))

    return auroc, f_score_max, threshold_best.item()

