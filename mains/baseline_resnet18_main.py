import os

from dataloader.classify_dataset import ClassifyDataset
import datetime
from torch.utils.data import DataLoader
import constants as const

import torch.nn as nn
import torch
import time
from tools.mylogging import Logger
import sys
from models.backbone_resnet18 import Resnet18
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
    checkpoint_dir = os.path.join(
        project_checkpoint_dir, f"exp{len(os.listdir(project_checkpoint_dir))}_resnet18_{datetime_str}"
    )
    os.makedirs(checkpoint_dir, exist_ok=True)

    project_save_dir = os.path.join(project_dir, const.SAVE_DIR)
    if not os.path.exists(project_save_dir):
        os.mkdir(project_save_dir)
    save_dir = os.path.join(
        project_save_dir,
        f"exp{len(os.listdir(project_checkpoint_dir))}_resnet18_{datetime_str}"
    )
    os.makedirs(save_dir, exist_ok=True)

    sys.stdout = Logger(os.path.join(save_dir,  'logs.txt'))

    # Create Visdom
    # vis = visdom.Visdom(env=opt.visdom_env)

    # Transform
    train_transformer = transforms.Compose([
        transforms.Resize((256, 256), interpolation=3),  # size=opt.height (224, 224)
        transforms.RandomHorizontalFlip(),
        # transforms.RandomCrop(size=opt.height),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    print('------------train_transformer------------')
    print(train_transformer)
    test_transformer = transforms.Compose([
        transforms.Resize((256, 256), interpolation=3),
        # transforms.CenterCrop(size=opt.height),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    print('------------test_transformer------------')
    print(test_transformer)

    # dataset
    train_data = ClassifyDataset("/home/log/PycharmProjects/fastflow_v2/datasets/classify_data_multi",
                                 mode="train", transform=train_transformer)
    test_data = ClassifyDataset("/home/log/PycharmProjects/fastflow_v2/datasets/classify_data_multi",
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
    model = Resnet18(num_train_classes)
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

        if test_acc > best_acc and epoch >= 5:
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
    print('Best at epoch %d, test accuracy %f' % (best_epoch, best_acc))


def _accuracy(model, test_lodaer, use_cuda=True):
    model.train(False)
    num_correct = 0.0
    num_total = 0
    for imgs, pids, img_names in test_lodaer:

        inputs = imgs.cuda()
        targets = pids.cuda()

        outputs = model(inputs)

        _, preds = torch.max(outputs.data, 1)
        num_total += pids.size(0)
        num_correct += torch.sum(preds == targets.data)
    model.train(True)
    accuracy = 100.0 * num_correct.item() / num_total
    return accuracy


if __name__ == '__main__':
    # import fire
    # fire.Fire()

    main()
