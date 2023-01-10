import os
import copy
from dataloader.classify_dataset import ClassifyDataset
import datetime
from torch.utils.data import DataLoader
import constants as const

import torch.nn as nn
import torch
import time
from tools.mylogging import Logger
import sys
from models.backbone_resnet50 import Resnet50
from models.focal_loss import FocalLoss
from torchvision import transforms

from torch.optim import lr_scheduler

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main(**kwargs):
    datetime_str = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M')
    project_checkpoint_dir = os.path.join(project_dir, const.CHECKPOINT_DIR)
    if not os.path.exists(project_checkpoint_dir):
        os.mkdir(project_checkpoint_dir)
    exp_dir_index = len(os.listdir(project_checkpoint_dir))
    checkpoint_dir = os.path.join(
        project_checkpoint_dir, f"exp{exp_dir_index}_resnet50_{datetime_str}"
    )
    os.makedirs(checkpoint_dir, exist_ok=True)

    project_save_dir = os.path.join(project_dir, const.SAVE_DIR)
    if not os.path.exists(project_save_dir):
        os.mkdir(project_save_dir)
    save_dir = os.path.join(
        project_save_dir,
        f"exp{exp_dir_index}_resnet50_{datetime_str}"
    )
    os.makedirs(save_dir, exist_ok=True)
    print(save_dir)

    sys.stdout = Logger(os.path.join(save_dir,  'logs.txt'))

    # Create Visdom
    # vis = visdom.Visdom(env=opt.visdom_env)

    # Transform
    train_transformer = transforms.Compose([
        # transforms.Resize((128, 128), interpolation=3),  # size=opt.height (224, 224)
        transforms.Resize((256, 256), interpolation=3),  # size=opt.height (224, 224)
        transforms.RandomHorizontalFlip(),
        # transforms.RandomCrop(size=128),
        transforms.RandomCrop(size=256),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    print('------------train_transformer------------')
    print(train_transformer)
    test_transformer = transforms.Compose([
        # transforms.Resize((128, 128), interpolation=3),
        transforms.Resize((256, 256), interpolation=3),
        # transforms.CenterCrop(size=opt.height),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    print('------------test_transformer------------')
    print(test_transformer)

    # dataset
    train_data = ClassifyDataset("/data/BYD_dingzi/dataset/duanziqiliu_manual_crop_classify_v3",
                                 mode="train", transform=train_transformer)
    test_data = ClassifyDataset("/data/BYD_dingzi/dataset/duanziqiliu_manual_crop_classify_v3",
                                mode="test", transform=test_transformer)
    num_train_classes = train_data.num_class

    # Create data loader
    print('load data')
    train_loader = DataLoader(
        train_data, batch_size=32, num_workers=4,
        shuffle=True,  pin_memory=True,  # drop_last=True,
    )
    test_lodaer = DataLoader(
        test_data, batch_size=32, num_workers=4,
        shuffle=False, pin_memory=True,
    )

    # Create model
    model = Resnet50(num_train_classes)
    model = nn.DataParallel(model).cuda()
    print('---------------model layers---------------')
    print(model)
    print('------------------end---------------------')

    # Criterion
    criterion = nn.CrossEntropyLoss()
    # focal loss
    focal_loss = FocalLoss(gamma=2)

    # Optimizer
    # features_param_ids = list(map(id, model.module.base_model.parameters()))
    # new_params = [p for p in model.parameters() if id(p) not in features_param_ids]
    # param_groups = [
    #     {'params': model.module.base_model.parameters(), 'lr': 0.01},
    #     {'params': new_params, 'lr': 0.1}
    # ]
    # optimizer = torch.optim.SGD(param_groups, lr=opt.lr, momentum=opt.momentum,
    #                             weight_decay=opt.weight_decay, nesterov=True)
    optimizer = torch.optim.SGD(model.module.parameters(), lr=0.01, momentum=0.9,
                                weight_decay=5e-4, nesterov=True)
    # Schedule for learning rate
    scheduler = lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.1, patience=6, verbose=True, threshold=1e-4)

    # define start time
    train_start_time = time.time()
    # define best acc
    best_acc = 60.0
    best_epoch = None

    # Start training
    for epoch in range(100):
        print('Epoch {}/{}'.format(epoch + 1, 100))

        model.train(True)

        # define loss in each epoch
        train_loss = 0.0
        # define the correct nums in each epoch
        num_total = 0
        num_correct = 0.0

        for datas in train_loader:
            imgs, pids, _ = datas
            current_batch_size, c, h, w = imgs.shape
            if current_batch_size == 1:  # if batch_size == 1, batch_normal can raise an error
                continue
            inputs = imgs.cuda()
            targets = pids.cuda()

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            outputs = model(inputs)
            # loss = criterion(outputs, targets)

            loss = focal_loss(outputs, targets)

            # Prediction
            _, preds = torch.max(outputs.data, 1)
            num_total += pids.size(0)
            # sum_ the correct nums in each batch_size
            num_correct += torch.sum(preds == targets.data)
            # sum_ the loss in each batch_size
            train_loss += loss.item()  # in the version 0.4.0

            # backward and optimize
            loss.backward()
            optimizer.step()

        # every epoch loss and accuracy in training
        epoch_loss = train_loss / num_total
        train_acc = 100.0 * num_correct.item() / num_total
        # compute the accuracy in test
        test_acc = _accuracy(model, test_lodaer)
        print('train Loss: {:.4f} Acc: {:.4f},  test Acc: {:.4f}'.format(
            epoch_loss, train_acc, test_acc,
        ))

        # adjust learning rate
        scheduler.step(test_acc)
        print('-' * 10)

        if test_acc >= best_acc and epoch >= 5:
            best_acc = test_acc
            best_epoch = epoch + 1
            # save model
            torch.save(model.module.state_dict(),
                       os.path.join(checkpoint_dir, 'epoch_' + str(epoch+1) + '_model.pth'))

        if epoch % 10 == 0:
            torch.save(model.module.state_dict(), os.path.join(checkpoint_dir, 'last_model.pth'))

    time_elapsed = time.time() - train_start_time
    print('Experiment complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60
    ))
    print(f'Best at epoch {best_epoch}, test accuracy {best_acc}')


def _accuracy(model, test_lodaer, use_cuda=True):
    model.train(False)
    class_info = {"correct_num": 0, "total_num": 0, "accuracy": 0.0}
    classes_count_dict = {"all": copy.deepcopy(class_info)}

    start = time.time()
    # if model_name == "resnet18_thread":
    #     _accuracy_resnet18_threshold(model, dataloader)
    for images, label_ids, img_names in test_lodaer:
        inputs = images.cuda()
        targets = label_ids.cuda()

        # outputs = model(inputs)
        outputs = model(inputs)

        soft_outs = torch.softmax(outputs.data, dim=1)
        # _, preds = torch.max(outputs_.data, 1)
        # _, preds = outputs_.data.topk(1, dim=1, largest=True)
        _, preds = soft_outs.data.topk(1, dim=1, largest=True)

        for index, item_outputs in enumerate(soft_outs):
            gt_label = targets[index].cpu().numpy()
            if str(gt_label) not in classes_count_dict:
                classes_count_dict[str(gt_label)] = copy.deepcopy(class_info)

            classes_count_dict["all"]["total_num"] += 1
            classes_count_dict[str(gt_label)]["total_num"] += 1

            pred = preds[index].cpu().numpy()

            if pred == gt_label:
                classes_count_dict["all"]["correct_num"] += 1
                classes_count_dict[str(gt_label)]["correct_num"] += 1

    end = time.time()
    print(f"total time cost: {end - start}, avg time: {(end - start) / classes_count_dict['all']['total_num']}")

    for k, v in classes_count_dict.items():
        v["accuracy"] = v["correct_num"] / v["total_num"]
        print(f"{k}: {v}")
    return classes_count_dict["all"]["accuracy"] * 100


if __name__ == '__main__':
    # import fire
    # fire.Fire()

    main()
