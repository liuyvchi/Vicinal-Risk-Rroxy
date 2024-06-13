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
from torchvision import transforms
import torchvision.models as models
import torch.utils.data as data
import torch.nn.functional as F

from dataset import RafDataset, MaEXDataset, AffectData, AffectData_twoTransforms, MS1MMaEX_Dataset, MS1MMaEXRec_Dataset, RafDataset_twoTransforms
from model import Model
from utils import *
# from resnet import *
from loss import ACLoss, convert_label_to_similarity, AU_Loss


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

class my_pool_manager():
    def __init__(self, args, ganimation_model, erase_loader):
        self.args = args
        self.imgs_pool = AUerased_imgs_pool(exchange_aus=True)
        self.ganimation_model = ganimation_model
        self.erase_loader = erase_loader
        self.erase_iterator = iter(erase_loader)
        self.hard_aus_pool = Queue()

    def put_imgs(self, imgs, aus, labels, wrong_predicts, FIRST_EPC):
        # while not self.imgs_pool.ISFULL:
        self.imgs_pool.FIRST_EPC = FIRST_EPC
        self.imgs_pool.put_imgs(self.ganimation_model, imgs, aus, labels, wrong_predicts, self.hard_aus_pool)

    def put_aus(self, aus, labels):
        self.hard_aus_pool.put({'aus':aus, 'labels':labels})



def print_time(threadName, delay, counter):
    while counter:
        if exitFlag:
            (threading.Thread).exit()
        time.sleep(delay)
        print
        "%s: %s" % (threadName, time.ctime(time.time()))
        counter -= 1


def train(args, pool_manager, model, train_loader, optimizer, scheduler, device, epoch_i):
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

    for batch_i, (raw_imgs, ganimation_imgs, real_aus, aus_valid, raw_labels, indexes) in enumerate(tqdm(train_loader)):

        raw_imgs = raw_imgs.to(device)
        real_aus = real_aus.to(device)
        raw_labels = raw_labels.to(device)

        raw_output, features = model(raw_imgs)
        _, raw_predicts = torch.topk(raw_output, k=2, dim=1)
        first_predicts = raw_predicts[:, 0]
        raw_correct_num = torch.eq(first_predicts, raw_labels).sum()
        wrong_index = torch.eq(first_predicts, raw_labels) == 0
        second_predicts = raw_predicts[:, 1]

        valid_features = features[aus_valid]
        valid_output = raw_output[aus_valid]
        valid_aus = real_aus[aus_valid]
        valid_aus_label = raw_labels[aus_valid]

        sp_out, sn_out = convert_label_to_similarity(valid_features, valid_aus_label)
        sp_aus, sn_aus = convert_label_to_similarity( nn.functional.normalize(valid_aus), valid_aus_label)
        # auloss = AU_Loss()(sp_out, sn_out, sp_aus, sn_aus)


        # positive_mask =  torch.eq(raw_labels, raw_labels.flip(0)) == 1
        # negative_mask =  torch.eq(raw_labels, raw_labels.flip(0)) == 0
        # pos_logits_d = nn.PairwiseDistance(p=2)(raw_predicts[positive_mask], raw_predicts.flip(0)[positive_mask])
        # neg_logits_d = nn.PairwiseDistance(p=2)(raw_predicts[negative_mask], raw_predicts.flip(0)[negative_mask])
        pos_logits_sum += torch.mean(sp_out)
        neg_logits_sum += torch.mean(sn_out)


        # positive_mask_aus = torch.eq(valid_aus_label, valid_aus_label.flip(0)) == 1
        # negative_mask_aus = torch.eq(valid_aus_label, valid_aus_label.flip(0)) == 0
        # pos_aus_d = nn.PairwiseDistance(p=2)(valid_aus[positive_mask_aus], valid_aus.flip(0)[positive_mask_aus])
        # neg_aus_d = nn.PairwiseDistance(p=2)(valid_aus[negative_mask_aus], valid_aus.flip(0)[negative_mask_aus])
        pos_aus_sum += torch.mean(sp_aus)
        neg_aus_sum += torch.mean(sn_aus)


        correct_sum += raw_correct_num
        sample_cnt += len(raw_labels)

        criterion = nn.CrossEntropyLoss()
        # loss = criterion(final_output, final_label)
        loss = criterion(raw_output, raw_labels)
        # loss = auloss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # assert (0)

        iter_cnt += 1
        running_loss += loss


    scheduler.step()
    running_loss = running_loss / iter_cnt
    acc = correct_sum.float() / float(sample_cnt)

    print(pos_logits_sum/iter_cnt, neg_logits_sum/iter_cnt, pos_aus_sum/iter_cnt, neg_aus_sum/iter_cnt)

    return acc, running_loss


    
def test(model, test_loader, device):
    predicted = []
    gt = []
    with torch.no_grad():
        model.eval()

        running_loss = 0.0
        iter_cnt = 0
        correct_sum = 0
        data_num = 0

        for batch_i, (imgs, labels, indexes) in enumerate(test_loader):
            imgs = imgs.to(device)
            labels = labels.to(device)

            outputs, _ = model(imgs)

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

        ## confusion matrix
        predicted = torch.cat(predicted, dim=-1).numpy()
        gt = torch.cat(gt, dim=-1).numpy()
        print(metrics.classification_report(gt, predicted))

    return test_acc, running_loss
        
        
        
def main(args):
    setup_seed(0)

    # torch.cuda.set_device(args.local_rank)
    train_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(scale=(0.02, 0.25)) ])
    
    eval_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])])

    ganimation_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize([128, 128]),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])


    # train_dataset = RafDataset(args, phase='train', transform=train_transforms)
    train_dataset = AffectData_twoTransforms(args, phase='train', transform=train_transforms, \
                                             transform2=ganimation_transforms)
    test_dataset = RafDataset_twoTransforms(args, phase='test', transform=eval_transforms)
    erase_dataset = RafDataset(args, phase='train', transform=ganimation_transforms)

    # train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)


    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.batch_size_fer,
                                               shuffle=True,
                                               num_workers=args.workers,
                                               pin_memory=True,
                                               drop_last=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size_fer,
                                              shuffle=False,
                                              num_workers=args.workers,
                                              pin_memory=True)
    erase_loader = torch.utils.data.DataLoader(erase_dataset, batch_size=args.batch_size,
                                              shuffle=True,
                                              num_workers=args.n_threads,
                                              pin_memory=True)

    
    model = Model(args).cuda()
    # local_rank = int(os.environ["LOCAL_RANK"])
    # model_ganimation = torch.nn.parallel.DistributedDataParallel(model_ganimation, device_ids=[local_rank], output_device=local_rank)
    model = torch.nn.DataParallel(model, device_ids=args.gpu_ids)
    device = torch.device('cuda:0')

    # checkpoint = torch.load('log/R18MS1M_raf_AUexchange/epoch_20_checkpoint.pth')
    # model.module.load_state_dict(checkpoint, strict=True)

    ganimation_model = create_model(args)

    optimizer\
         = torch.optim.Adam(params=[{"params": model.parameters()}], lr=0.0001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)


    # test_acc, test_loss = test(model_ganimation, test_loader, device)
    # print(test_acc, test_loss)
    # assert (0)

    # log_dir = "log/R18MS1M_raf_AU_mixupHardAUs"
    log_dir = "log/R18MS1M_raf_AUsLogitsRelation"
    args.use_raw_imgs = False
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)

    pool_manager = my_pool_manager(args, ganimation_model, erase_loader)
    # pool_manager.start()

    for i in range(1, args.epochs + 1):
        train_acc, train_loss = train(args, pool_manager, model, train_loader, optimizer, scheduler, device, i)
        test_acc, test_loss = test(model, test_loader, device)
        print(i, train_acc, train_loss)
        print(i, test_acc, test_loss)
        with open( os.path.join(log_dir, 'log.txt'), 'a') as f:
            f.write(str(i)+'_'+str(test_acc)+'_'+str(test_loss)+'_'+str(train_acc)+'_'+str(train_loss)+'\n')
        if i > (args.epochs - 30):
            torch.save(model.module.state_dict(), os.path.join(log_dir, ('epoch_%s_checkpoint.pth' % i)))
    # for i in range(1, args.epochs + 1):
    #     train_acc, train_loss = train(args, model_ganimation, train_loader2, optimizer, scheduler, device)
    #     test_acc, test_loss = test(model_ganimation, test_loader, device)
    #     print(i, test_acc, test_loss)
    #     with open('log/IR50_scratch.txt', 'a') as f:
    #         f.write(str(i)+'_'+str(test_acc)+'\n')


if __name__ == '__main__':
    opt = Options().parse()
    main(opt)
