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
from copy import deepcopy

import torch
import torch.nn as nn
import torchvision
import torch.optim as optim
from torchvision import transforms, datasets
import torchvision.models as models
import torch.utils.data as data
import torch.nn.functional as F
import timm

from dataset import RafDataset, MaEXDataset, AffectData, AffectData_twoTransforms, MS1MMaEX_Dataset, MS1MMaEXRec_Dataset, RafDataset_twoTransforms, ImageNetV, ImageNetV_twoTransforms, cifar10_twoTransforms, cifar10_C_twoTransforms, cifar10_v1_twoTransforms
from model import Model
from utils import *
# from resnet import *
from loss import ACLoss, convert_label_to_AUsim, AU_Loss, convert_label_to_EI, convert_label_to_overlap, compute_vicinalRisk_L2, test_tprATfpr, roc_auc, convert_label_to_intensDis, mixup_data, mixup_criterion

from ImageNetAll import imagenet_a_mask, imagenet_r_mask, imagenet_o_mask
imagenet_mask = imagenet_o_mask

import models
import pytorch_cifar_models

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

##DoGAN
import DaGAN.modules.generator as GEN
from DaGAN.sync_batchnorm import DataParallelWithCallback
import DaGAN.depth as depth
from DaGAN.modules.keypoint_detector import KPDetector
import yaml
from collections import OrderedDict
from torchvision.transforms.functional import InterpolationMode
import torchvision.transforms.functional as TF
import random

class MyRotationTransform:
    """Rotate by one of the given angles."""

    def __init__(self, angles):
        self.angles = angles

    def __call__(self, x):
        angle = random.choice(self.angles)
        return TF.rotate(x, angle)


def get_mixup_predictions(model, test_loader, device):
    results = {}

    with torch.no_grad():
        model.eval()

        running_loss = 0.0
        iter_cnt = 0
        correct_sum = 0
        data_num = 0

        seed = 10
        torch.manual_seed(seed)
        for batch_i, (imgs1, imgs2, raw_labels, index) in enumerate(test_loader):

            imgs1 = imgs1.to(device)
            raw_labels = raw_labels.to(device)

            # output1 = model(imgs1)[:,imagenet_mask]
            output1 = model(imgs1)
            iter_cnt += 1
            _, predicts = torch.max(output1, 1)

            imgs2, targets_a, targets_b, lam, mix_index = mixup_data(imgs1, predicts, batch_i)


            # output2 = model(imgs2)[:,imagenet_mask]
            output2 = model(imgs2)

            correct_num = torch.eq(predicts, raw_labels).sum()
            correct_sum += correct_num
            data_num += len(raw_labels)

            for i in range(len(index)):
                if str(index[i]) not in results.keys():
                    results[str(index[i])] = [output1[i].cpu().numpy(), output2[i].cpu().numpy(), lam[i].cpu().numpy(), index[mix_index[i]]]
                    
            del output1
            del output2

        test_acc = correct_sum.float() / float(data_num)
        results['acc'] = test_acc

    return results


def get_grey_predictions(model, test_loader, device):
    results = {}

    with torch.no_grad():
        model.eval()

        running_loss = 0.0
        iter_cnt = 0
        correct_sum = 0
        data_num = 0
        count = 0

        seed = 10
        torch.manual_seed(seed)
        for batch_i, (imgs1, imgs2, raw_labels, index) in enumerate(test_loader):
        # # 单独适配cifar10测试集
        # for batch_i, (imgs1, raw_labels) in enumerate(test_loader):

            imgs1 = imgs1.to(device)
            raw_labels = raw_labels.to(device)

            # #### cifar10 插件， 需要及时删除 ！！！ 
            # imgs2 = imgs1
            # #####

            # output1 = model(imgs1)[:,imagenet_mask]
            output1 = model(imgs1)
            iter_cnt += 1
            _, predicts = torch.max(output1, 1)

            # output2 = model(imgs2)[:,imagenet_mask]
            output2 = model(imgs2)

            correct_num = torch.eq(predicts, raw_labels).sum()
            correct_sum += correct_num
            data_num += len(raw_labels)

            for i in range(len(raw_labels)):
                if str(count+i) not in results.keys():
                    results[str(count+i)] = [output1[i].cpu().numpy(), output2[i].cpu().numpy()]
            count+=len(raw_labels)

            # for i in range(len(index)):
            #     if str(index[i]) not in results.keys():
            #         results[str(index[i])] = [output1[i].cpu().numpy(), output2[i].cpu().numpy()]
                    
            del output1
            del output2

        test_acc = correct_sum.float() / float(data_num)
        results['acc'] = test_acc

    return results


def main(args):
    setup_seed(0)

    model_name = args.model_name
    job_id = args.job_id
    print(model_name)

    # mixup_prediction_path1 = '/root/paddlejob/workspace/env_run/afs/liuyuchi/autoEval/modelOutput/imagenetv2_a_out_mixup/%s.npy'%(model_name)
    # grey_prediction_path1 = '/root/paddlejob/workspace/env_run/afs/liuyuchi/autoEval/modelOutput/imagenetv2_a_out_grey/%s.npy'%(model_name)
    rotation_prediction_path1 = '/root/paddlejob/workspace/env_run/afs/liuyuchi/autoEval/modelOutput/cifar10_GaussianBlur3_test_out_rotation/%s.npy'%(model_name)


    if not os.path.exists(rotation_prediction_path1):
        pass
    else:
        # print('predictions have already been saved in '+mixup_prediction_path1)
        print('predictions have already been saved in '+rotation_prediction_path1)
        assert(0)

    
    
    # define models
    if 'cifar10_' in model_name:
        model = pytorch_cifar_models.create_model(model_name).cuda()
        checkpoint = torch.load('/root/paddlejob/workspace/env_run/afs/liuyuchi/pytorch_cifar_models_weights/%s.pt'%(model_name))
    elif '_20' in model_name[-3:]:
        model = models.create_model(model_name[:-3]).cuda()
        checkpoint = torch.load('/root/paddlejob/workspace/env_run/afs/liuyuchi/cifar_weights/cifar10_%s/epoch_20_checkpoint.pth'%(model_name[:-3]))
    else: 
        model = models.create_model(model_name).cuda()
        checkpoint = torch.load('/root/paddlejob/workspace/env_run/afs/liuyuchi/cifar_weights/cifar10_%s/epoch_50_checkpoint.pth'%(model_name))
        
    model = torch.nn.DataParallel(model, device_ids=args.gpu_ids)
    
    model.module.load_state_dict(checkpoint, strict=True)

    device = torch.device('cuda:0')


    # 创建测试集 DataLoader
    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2023, 0.1994, 0.2010]
    eval_transforms = transforms.Compose([transforms.ToPILImage(), transforms.Resize((32, 32)), transforms.ToTensor(), transforms.Normalize(mean, std)])
    # wjscore_transforms = transforms.Compose([transforms.ToPILImage(), transforms.Resize((32, 32)), transforms.ToTensor(), transforms.Grayscale(num_output_channels=3), transforms.Normalize(mean, std)])
    wjscore_transforms = transforms.Compose([transforms.ToPILImage(), transforms.Resize((32, 32)), transforms.ToTensor(), MyRotationTransform(angles=[-90, 90, 180]), transforms.Normalize(mean, std)])
                                   
    wj_test_dataset = cifar10_C_twoTransforms('../../data/CIFAR-10-C', transform=eval_transforms, transform2=wjscore_transforms)      
    wj_test_loader = torch.utils.data.DataLoader(wj_test_dataset, batch_size=args.batch_size_fer,
                                              shuffle=True,
                                              num_workers=args.workers,
                                              pin_memory=True)

    # # 单独适配cifar10测试集 
    # transform_test = transforms.Compose([
    #     transforms.Resize((32, 32)),
    #     transforms.ToTensor(),
    #     # MyRotationTransform(angles=[-90, 90, 180]),
    #     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    # ])
    # testset = torchvision.datasets.CIFAR10(
    #     root='/root/paddlejob/workspace/env_run/data/', train=False, download=False, transform=transform_test)
    # wj_test_loader = torch.utils.data.DataLoader(
    #     testset, batch_size=512, shuffle=False, num_workers=16)

    # log_dir = "log/R18MS1M_raf_AU_mixupHardAUs"
    
    log_dir = os.path.join("/root/paddlejob/workspace/env_run/output/", "cifar10_GaussianBlur3_test_out_rotation")
    args.use_raw_imgs = False
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)

    # pool_manager = my_pool_manager(args, ganimation_model, erase_loader)
    # pool_manager.start()

    # if os.path.exists(os.path.join(log_dir, 'log.txt')):
    #     os.remove(os.path.join(log_dir, 'log.txt')) 
    
    start_time = time.time()
    results_dic = get_grey_predictions(model, wj_test_loader, device)
    print("--- %s seconds ---" % (time.time() - start_time))
    print(results_dic['acc'])

    np.save(os.path.join(log_dir ,model_name+'.npy'), results_dic)
    

opt = Options().parse()
main(opt)