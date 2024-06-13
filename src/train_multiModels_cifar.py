# -*- coding: utf-8 -*-

import os
import cv2
import csv
import math
import random
import numpy as np
import pandas as pd
import argparse
import pickle
from tqdm import tqdm
from sklearn import datasets, svm, metrics

import torch
import torch.nn as nn
import torchvision
import torch.optim as optim
from torchvision import transforms
import torchvision.models as models
import torch.utils.data as data
import torch.nn.functional as F
import timm

from dataset import RafDataset, MaEXDataset, AffectData, AffectData_twoTransforms, MS1MMaEX_Dataset, MS1MMaEXRec_Dataset, RafDataset_twoTransforms
from model import Model
from utils import *
from model_names import model_names
# from resnet import *
from loss import ACLoss, convert_label_to_AUsim, AU_Loss



# import torch.distributed as dist
# dist.init_process_group(backend='gloo|nccl')

## generation packages
from model_ganimation import create_model
from options import Options
from AUerase_utils import AUerased_imgs_pool
import threading
import time
import logging
from queue import Queue

from models import *


def print_time(threadName, delay, counter):
    while counter:
        if exitFlag:
            (threading.Thread).exit()
        time.sleep(delay)
        print
        "%s: %s" % (threadName, time.ctime(time.time()))
        counter -= 1


def train(args, model, train_loader, optimizer, scheduler, device, epoch_i):
    running_loss = 0.0
    iter_cnt = 0
    correct_sum = 0
    sample_cnt = 0
    pos_logits_sum = 0
    neg_logits_sum = 0
    pos_aus_sum = 0
    neg_aus_sum = 0
    use_raw_imgs = args.use_raw_imgs
    FIRST_EPC = epoch_i == 1

    model.to(device)
    model.train()

    invariance_sum = 0


    for batch_i, (raw_imgs, labels) in enumerate(tqdm(train_loader)):
        raw_imgs = raw_imgs.to(device)
        labels = labels.to(device)

        criterion = nn.CrossEntropyLoss()
        raw_output = model(raw_imgs)
        loss = nn.CrossEntropyLoss()(raw_output, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        iter_cnt += 1
        _, predicts = torch.max(raw_output, 1)
        correct_num = torch.eq(predicts, labels).sum()
        correct_sum += correct_num
        running_loss += loss
        sample_cnt += raw_output.size(0)

    scheduler.step()
    running_loss = running_loss / iter_cnt
    acc = correct_sum.float() / float(sample_cnt)

    return acc.item(), running_loss.item()


    
def test(model, test_loader, device):
    predicted = []
    gt = []
    with torch.no_grad():
        model.eval()

        running_loss = 0.0
        iter_cnt = 0
        correct_sum = 0
        data_num = 0

        for batch_i, (raw_imgs, raw_labels) in enumerate(test_loader):
            imgs = raw_imgs.to(device)
            labels = raw_labels.to(device)

            outputs = model(imgs)

            loss = nn.CrossEntropyLoss()(outputs, labels)

            iter_cnt += 1
            _, predicts = torch.max(outputs, 1)

            predicted.append(predicts.cpu().detach())
            gt.append(labels.cpu().detach())

            correct_num = torch.eq(predicts, labels).sum()
            correct_sum += correct_num

            running_loss += loss
            data_num += outputs.size(0)

        running_loss = running_loss / iter_cnt
        test_acc = correct_sum.float() / float(data_num)

        # ## confusion matrix
        # predicted = torch.cat(predicted, dim=-1).numpy()
        # gt = torch.cat(gt, dim=-1).numpy()
        # print(metrics.classification_report(gt, predicted))

    return test_acc, running_loss
        
        
        
def main(args):
    setup_seed(0)

        # Data
    print('==> Preparing data..')
    # mean = [0.485, 0.456, 0.406]
    # std = [0.229, 0.224, 0.225]
    
    # 0.4914, 0.4822, 0.4465
    # 0.2023, 0.1994, 0.2010
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(
        root='/root/paddlejob/workspace/env_run/data/', train=True, download=False, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=512, shuffle=True, num_workers=16)

    testset = torchvision.datasets.CIFAR10(
        root='/root/paddlejob/workspace/env_run/data/', train=False, download=False, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=512, shuffle=False, num_workers=16)

    classes = ('plane', 'car', 'bird', 'cat', 'deer',
            'dog', 'frog', 'horse', 'ship', 'truck')

    # model = Model(args).cuda()
    # local_rank = int(os.environ["LOCAL_RANK"])
    # model_ganimation = torch.nn.parallel.DistributedDataParallel(model_ganimation, device_ids=[local_rank], output_device=local_rank)
    model_name = args.model_name
    print(model_name)
    model = create_model(model_name)
    model = torch.nn.DataParallel(model, device_ids=args.gpu_ids)
    device = torch.device('cuda:0')

    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

    log_dir = "/root/paddlejob/workspace/env_run/output/cifar10_"+model_name
    args.use_raw_imgs = False
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)

    for i in range(1, 50 + 1):
        train_acc, train_loss = train(args, model, trainloader, optimizer, scheduler, device, i)
        if i > 10 and i % 10 == 0:
            torch.save(model.module.state_dict(), os.path.join(log_dir, ('epoch_%s_checkpoint.pth' % i)))
            test_acc, test_loss = test(model, testloader, device)
            print(i, train_acc, train_loss)
            print(i, test_acc, test_loss)
            with open( os.path.join(log_dir, 'log.txt'), 'a') as f:
                f.write(str(i)+'_'+str(test_acc)+'_'+str(test_loss)+'_'+str(train_acc)+'_'+str(train_loss)+'\n')


if __name__ == '__main__':
    opt = Options().parse()
    main(opt)