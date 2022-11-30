import argparse
import os
import sys

import torch
import yaml
from ignite.contrib import metrics
import datetime
from tools.mylogging import Logger

import constants as const
import dataset
import fastflow
import utils

from postprocessing.caculate import get_best_thredhold
from postprocessing.plot import visualize_heatmap


def build_train_data_loader(args, config):
    train_dataset = dataset.MVTecDataset(
        root=args.data,
        category=args.category,
        input_size=config["input_size"],
        is_train=True,
    )
    return torch.utils.data.DataLoader(
        train_dataset,
        batch_size=const.BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        drop_last=True,
    )


def build_test_data_loader(args, config):
    test_dataset = dataset.MVTecDataset(
        root=args.data,
        category=args.category,
        input_size=config["input_size"],
        is_train=False,
    )
    return torch.utils.data.DataLoader(
        test_dataset,
        batch_size=const.BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )


def build_model(config):
    model = fastflow.FastFlow(
        backbone_name=config["backbone_name"],
        flow_steps=config["flow_step"],
        input_size=config["input_size"],
        conv3x3_only=config["conv3x3_only"],
        hidden_ratio=config["hidden_ratio"],
    )
    print(
        "Model A.D. Param#: {}".format(
            sum(p.numel() for p in model.parameters() if p.requires_grad)
        )
    )
    return model


def build_optimizer(params, optim_name="Adam"):
    if optim_name == "Adam":
        return torch.optim.Adam(
            params, lr=const.LR, weight_decay=const.WEIGHT_DECAY
        )
    elif optim_name == "SGD":
        return torch.optim.SGD(params, lr=const.LR, momentum=const.MOMENTUM,
                               weight_decay=const.WEIGHT_DECAY, nesterov=True)


def train_one_epoch(dataloader, model, optimizer, epoch):
    model.train()
    loss_meter = utils.AverageMeter()
    for step, data in enumerate(dataloader):
        # forward
        data = data.cuda()
        ret = model(data)
        loss = ret["loss"]
        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # log
        loss_meter.update(loss.item())
        if (step + 1) % const.LOG_INTERVAL == 0 or (step + 1) == len(dataloader):
            print(
                "Epoch {} - Step {}: loss = {:.3f}({:.3f})".format(
                    epoch + 1, step + 1, loss_meter.val, loss_meter.avg
                )
            )


def eval_once(dataloader, model, save_dir, best_auroc=0.0):

    model.eval()
    # auroc_metric = metrics.ROC_AUC()
    # for data, targets in dataloader:
    #     data, targets = data.cuda(), targets.cuda()
    #     with torch.no_grad():
    #         ret = model(data)
    #     outputs = ret["anomaly_map"].cpu().detach()
    #     outputs = outputs.flatten()
    #     targets = targets.flatten()
    #     auroc_metric.update((outputs, targets.type(torch.int64)))
    # auroc = auroc_metric.compute()

    auroc, f_score_max, threshold_best = get_best_thredhold(model, dataloader)

    if auroc > best_auroc:
        visualize_heatmap(model, dataloader, save_dir, threshold_best)

    # print("AUROC: {}".format(auroc))
    return auroc, f_score_max, threshold_best


def train(args):
    os.makedirs(const.CHECKPOINT_DIR, exist_ok=True)
    checkpoint_dir = os.path.join(
        const.CHECKPOINT_DIR, "exp%d" % len(os.listdir(const.CHECKPOINT_DIR))
    )
    os.makedirs(checkpoint_dir, exist_ok=True)

    os.makedirs(const.SAVE_DIR, exist_ok=True)
    save_dir = os.path.join(
        const.SAVE_DIR, f"exp{len(os.listdir(const.SAVE_DIR))}_{args.category}_{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M')}"
    )
    # Redirect print to both console and logs file
    sys.stdout = Logger(os.path.join(save_dir,  'logs.txt'))

    config = yaml.safe_load(open(args.config, "r"))
    model = build_model(config)

    features_param_ids = list(map(id, model.feature_extractor.parameters()))
    fast_flow_params = [p for p in model.parameters() if id(p) not in features_param_ids]
    param_groups = [
        {'params': model.feature_extractor.parameters(), 'lr': const.BACKBONE_LR},
        {'params': fast_flow_params, 'lr': const.LR}
    ]
    # optimizer = build_optimizer(model.parameters())
    optimizer = build_optimizer(param_groups)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.1, patience=6, verbose=True, threshold=1e-4)

    train_dataloader = build_train_data_loader(args, config)
    test_dataloader = build_test_data_loader(args, config)
    model.cuda()
    best_auroc = 0.0
    best_info = {}

    first_change = True

    for epoch in range(const.NUM_EPOCHS):
        print(f"Epoch:{epoch}  backbone_Lr:{optimizer.state_dict()['param_groups'][0]['lr']:.2E}, "
              f"fast_flow_Lr:{optimizer.state_dict()['param_groups'][1]['lr']:.2E}")

        if const.TRAINING_BACKBONE and epoch == const.NUM_EPOCHS // 2 and first_change:
            print(f"-------------------change module learning strategy---------------------")
            model.training_backbone = True
            model.change_params_requires_grad()
            first_change = False

        train_one_epoch(train_dataloader, model, optimizer, epoch)
        # if (epoch + 1) % const.EVAL_INTERVAL == 0:
        auroc, f_score_max, threshold_best = eval_once(test_dataloader, model, save_dir, best_auroc)
        if auroc > best_auroc:
            best_auroc = auroc
            best_info = {'auroc': auroc, 'f_score_max': f_score_max, "threshold_best": threshold_best}
        if (epoch + 1) % const.CHECKPOINT_INTERVAL == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                },
                os.path.join(checkpoint_dir, "%d.pt" % epoch),
            )
        if epoch >= const.NUM_EPOCHS // 2:
            scheduler.step(auroc)

    print(f"best_info: {best_info}")


def evaluate(args):

    os.makedirs(const.SAVE_DIR, exist_ok=True)
    save_dir = os.path.join(
        const.SAVE_DIR, "exp%d" % len(os.listdir(const.SAVE_DIR))
    )
    os.makedirs(save_dir, exist_ok=True)

    config = yaml.safe_load(open(args.config, "r"))
    model = build_model(config)
    checkpoint = torch.load(args.checkpoint)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_dataloader = build_test_data_loader(args, config)
    model.cuda()
    eval_once(test_dataloader, model, save_dir)


def parse_args():
    parser = argparse.ArgumentParser(description="Train FastFlow on MVTec-AD dataset")
    parser.add_argument(
        "-cfg", "--config", default='configs/resnet18.yaml', type=str,  help="path to config file",  # required=True,
    )
    parser.add_argument("--data", type=str, default='datasets/MVTec', help="path to mvtec folder",
                        # required=True,
                        )
    parser.add_argument(
        "-cat",
        "--category",
        default='amz_1_down',
        type=str,
        choices=const.MVTEC_CATEGORIES,
        # required=True,
        help="category name in mvtec",
    )
    parser.add_argument("--eval", action="store_true", help="run eval only")
    parser.add_argument(
        "-ckpt", "--checkpoint", type=str, help="path to load checkpoint"
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    if args.eval:
        evaluate(args)
    else:
        train(args)
