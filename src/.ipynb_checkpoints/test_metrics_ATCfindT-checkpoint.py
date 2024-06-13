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
from torchvision import transforms, datasets
import torchvision.models as models
import torch.utils.data as data
import torch.nn.functional as F
import timm

from dataset import RafDataset, MaEXDataset, AffectData, AffectData_twoTransforms, MS1MMaEX_Dataset, MS1MMaEXRec_Dataset, RafDataset_twoTransforms, ImageNetV, ImageNetV_twoTransforms, ImageNetPredictions
from model import Model
from utils import *
# from resnet import *
from loss import ACLoss, convert_label_to_AUsim, AU_Loss, convert_label_to_EI, convert_label_to_overlap, convert_label_to_overlapMatrix, convert_label_to_cdistMatrix, gaussian_kernal,convert_label_to_confMatrix, compute_vicinalRisk_L2, test_tprATfpr, roc_auc, convert_label_to_intensDis, mixup_data, mixup_criterion

from backbones import get_model

from ImageNetAll import imagenet_a_mask, imagenet_r_mask
imagenet_mask = imagenet_r_mask



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

def test_wj(test_loader, device):

    predicted = []
    gt = []
    with torch.no_grad():

        running_loss = 0.0
        iter_cnt = 0
        correct_sum = 0
        data_num = 0

        ver_correct_sum = [0 for i in range(9)]
        ver_p_ausT = [1-0.1*i for i in range(10)]
        ver_n_ausT = [0.1*i for i in range(10)]
        pairs_index = [[] for i in range(9)]
        ver_num = [0 for i in range(9)]
        ver_acc_hard = [0 for i in range(9)]
        
        ver_score = []
        ver_label = []

        wj_score_sum=0
        I_overlap_score_sum = 0
        pnI_overlap_score_sum = 0
        ausP_EI_sum = 0
        ausN_EI_sum = 0
        EI_randomPair_sum = 0
        EI_halfvrm = 0
        EI_vrm_1 = 0
        EI_vrm_2 = 0
        EI_vrm_all = 0
        grey_vrm_sum = 0
        grey_vrmV2_sum = 0
        grey_vrmV3_sum = 0
        rota_vrm_sum = 0
        ensemble_agreement = 0
        vrm_pairs_1 = 0
        vrm_pairs_2 = 0
        vrm_pairs_all = 0
        d11, d12, d21, d22= 0, 0, 0, 0
        sd11, sd12, sd21, sd22= 0, 0, 0, 0

        vicinity1 = []
        vicinity2 = []
        vicinity_predictions1 = []
        vicinity_predictions2 = []
        vicinity_confidence1 = []
        vicinity_confidence2 = []

        AC = 0
        DoC= 0
        ATC = 0
        

        seed = 10
        torch.manual_seed(seed)
        T_value = 0
        for batch_i, (prediction1_1, prediction1_2, prediction2_1, prediction2_2, acc, gt) in enumerate(test_loader):
            output1_1 = prediction1_1.to(device)
            output1_2 = prediction1_2.to(device)
            softmax_out1 =  F.softmax(output1_1, dim=-1)
            softmax_out2 = F.softmax(output1_2, dim=-1)

            _, predicts1 = torch.topk(output1_1, k=2, dim=1)
            _, predicts2 = torch.topk(output1_2, k=2, dim=1)
            first_predicts1 = predicts1[:, 0]
            first_predicts2 = predicts2[:, 0]
            range_index = torch.tensor(range(len(softmax_out1)))
            confidence_1 = softmax_out1[range_index, first_predicts1]
            confidence_2 = softmax_out2[range_index, first_predicts2]

            vicinity1.append(softmax_out1)
            vicinity2.append(softmax_out2)
            vicinity_predictions1.append(first_predicts1)
            vicinity_predictions2.append(first_predicts2)
            vicinity_confidence1.append(confidence_1)
            vicinity_confidence2.append(confidence_2)
        
        vicinity1=torch.cat(vicinity1, dim=0)
        vicinity2=torch.cat(vicinity2, dim=0)
        vicinity_predictions1=torch.cat(vicinity_predictions1, dim=0)
        vicinity_predictions2=torch.cat(vicinity_predictions2, dim=0)
        vicinity_confidence1 = torch.cat(vicinity_confidence1, dim=0)
        vicinity_confidence2 = torch.cat(vicinity_confidence2, dim=0)
        
        source_accuracy = acc[0].item()
        print('source_accuracy', source_accuracy)

        grid_first = np.linspace(1, 0, 11)
        for i in grid_first:
            acc_p = (vicinity_confidence1>i).sum() / len(vicinity_confidence1)
            if acc_p > source_accuracy:
                grid_second = np.linspace(i, i+0.1, 11)
                print("second search", grid_second)
                for j in grid_second:
                    acc_p = (vicinity_confidence1>j).sum() / len(vicinity_confidence1)
                    if acc_p <=source_accuracy:
                        print("find T", j)
                        T_value = j
                        break
                break
            else:
                continue


    return acc[0].item(), T_value


def main(args):
    setup_seed(0)

    model_name = args.model_name
    job_id = args.job_id
    print(model_name)
    
    ## iwildcam_erm_tuneE_seed0 & iwildcam_erm_tuneF_seed0
    if model_name == 'ResNet50':
        model2_name = 'ResNet34'
    else:
        model2_name = 'ResNet50'

    grey_prediction_path1 = '/root/paddlejob/workspace/env_run/afs/liuyuchi/autoEval/modelOutput/cifar10_test_out_rotation/%s.npy'%(model_name)
    grey_prediction_path2 = '/root/paddlejob/workspace/env_run/afs/liuyuchi/autoEval/modelOutput/cifar10_test_out_rotation/%s.npy'%(model2_name)

                    
    wj_test_dataset = ImageNetPredictions(args.ImageNetV2_path, grey_prediction_path1, grey_prediction_path2, type='grey')      
                                   
    wj_test_loader = torch.utils.data.DataLoader(wj_test_dataset, batch_size=args.batch_size_fer,
                                              shuffle=True,
                                              num_workers=args.workers,
                                              pin_memory=True,
                                              drop_last=False)
    
    log_dir = os.path.join("/root/paddlejob/workspace/env_run/output/", "cifar10_test_ATC")
    args.use_raw_imgs = False
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)

    # pool_manager = my_pool_manager(args, ganimation_model, erase_loader)
    # pool_manager.start()

    # if os.path.exists(os.path.join(log_dir, 'log.txt')):
    #     os.remove(os.path.join(log_dir, 'log.txt')) 
    device = torch.device('cuda:0')
    start_time = time.time()

    test_acc, Threshold = test_wj(wj_test_loader, device)
    print("--- %s seconds ---" % (time.time() - start_time))

    # print(i, train_acc, train_loss)
    dic = {}
    dic['test_acc'] = test_acc
    # dic['test_loss'] = test_loss
    # dic['confidence'] = confidence_mean

    dic['Threshold'] = Threshold

    print(test_acc, Threshold)
    np.save(os.path.join(log_dir ,model_name+'.npy'), dic)
    

opt = Options().parse()
main(opt)