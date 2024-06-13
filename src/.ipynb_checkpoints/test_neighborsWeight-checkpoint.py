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

from dataset import RafDataset, MaEXDataset, AffectData, AffectData_twoTransforms, MS1MMaEX_Dataset, MS1MMaEXRec_Dataset, RafDataset_twoTransforms, ImageNetV, ImageNetV_twoTransforms, ImageNetPredictions_gt
from model import Model
from utils import *
# from resnet import *
from loss import ACLoss, convert_label_to_AUsim, AU_Loss, convert_label_to_EI, convert_label_to_overlap, convert_label_to_overlapMatrix, convert_label_to_cdistMatrix, gaussian_kernal, convert_label_to_confMatrix, compute_vicinalRisk_L2, test_tprATfpr, roc_auc, convert_label_to_intensDis, mixup_data, mixup_criterion

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

def test_wj(test_loader, exter_knowledge_dic, device):

    S_AC = exter_knowledge_dic['S_AC']
    S_accuracy = exter_knowledge_dic['S_accuracy']
    S_ATC_t = exter_knowledge_dic['ATC_t']

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
        AC = 0
        AC_vrp_sum = 0
        predict_Inv = 0 
        predict_Inv_vrp_sum = 0 
        confidence_Inv = 0
        confidence_Inv_vrp_sum = 0
        DoC_acc = 0
        DoC_vrp_acc = 0 
        ATC_acc = 0
        ATC_vrp_acc_sum = 0
        
        vicinity1_tmp = []
        vicinity2_tmp = []
        vicinity_predictions1_tmp = []
        vicinity_predictions2_tmp = []
        vicinity_confidence1_tmp = []
        vicinity_confidence2_tmp = []
        index_tmp = []
        vicinity_gt_tmp = []
    
        correct_weighVSacc_correctNum = [0 for i in range(10)]
        correct_weighVSacc_totalNum = [0 for i in range(10)]
        incorrect_weighVSacc_correctNum = [0 for i in range(10)]
        incorrect_weighVSacc_totalNum = [0 for i in range(10)]

        correct_weighVSacc = [0 for i in range(10)]
        incorrect_weighVSacc = [0 for i in range(10)]

        seed = 10
        torch.manual_seed(seed)
        for batch_i, (prediction1_1, prediction1_2, prediction2_1, prediction2_2, acc, gt, idx) in enumerate(test_loader):
            output1_1 = prediction1_1.to(device)
            output1_2 = prediction1_2.to(device)
            softmax_out1 =  F.softmax(output1_1, dim=-1)
            softmax_out2 = F.softmax(output1_2, dim=-1)
            idx = idx.to(device)
            gt = gt.to(device)

            _, predicts1 = torch.topk(output1_1, k=2, dim=1)
            _, predicts2 = torch.topk(output1_2, k=2, dim=1)
            first_predicts1 = predicts1[:, 0]
            first_predicts2 = predicts2[:, 0]
            range_index = torch.tensor(range(len(softmax_out1)))
            confidence_1 = softmax_out1[range_index, first_predicts1]
            confidence_2 = softmax_out2[range_index, first_predicts2]

            vicinity1_tmp.append(softmax_out1)
            vicinity2_tmp.append(softmax_out2)
            vicinity_predictions1_tmp.append(first_predicts1)
            vicinity_predictions2_tmp.append(first_predicts2)
            vicinity_confidence1_tmp.append(confidence_1)
            vicinity_confidence2_tmp.append(confidence_2)
            index_tmp.append(idx)
            vicinity_gt_tmp.append(gt)

        index_tmp = torch.cat(index_tmp, dim=0)
        length = len(index_tmp)
        # print(length)
        # assert(0)
        selected_num = 30000
        index_tmp = index_tmp[:selected_num]
        vicinity_gt_tmp = torch.cat(vicinity_gt_tmp, dim=0)[:selected_num]
        vicinity1_tmp = torch.cat(vicinity1_tmp, dim=0)[:selected_num]
        vicinity2_tmp = torch.cat(vicinity2_tmp, dim=0)[:selected_num]
        vicinity_predictions1_tmp = torch.cat(vicinity_predictions1_tmp, dim=0)[:selected_num]
        vicinity_predictions2_tmp = torch.cat(vicinity_predictions2_tmp, dim=0)[:selected_num]
        vicinity_confidence1_tmp = torch.cat(vicinity_confidence1_tmp, dim=0)[:selected_num]
        vicinity_confidence2_tmp = torch.cat(vicinity_confidence2_tmp, dim=0)[:selected_num]

        for batch_i, (prediction1_1, prediction1_2, prediction2_1, prediction2_2, acc, gt, idx) in enumerate(test_loader):

            output1_1 = prediction1_1.to(device)
            output1_2 = prediction1_2.to(device)
            # output2_1 = prediction2_1.to(device)
            # output2_2 = prediction2_2.to(device)
            softmax_out1 =  F.softmax(output1_1, dim=-1)
            softmax_out2 = F.softmax(output1_2, dim=-1)
            confidence1 = softmax_out1.max(dim=-1)[0]
            confidence2 = softmax_out2.max(dim=-1)[0]
            idx = idx.to(device)
            gt = gt.to(device)

            _, predicts1 = torch.topk(output1_1, k=2, dim=1)
            _, predicts2 = torch.topk(output1_2, k=2, dim=1)
            first_predicts1 = predicts1[:, 0]
            first_predicts2 = predicts2[:, 0]
            same_pred = torch.eq(first_predicts1, first_predicts2)
            notsame_pred = ~same_pred
            
            ## neighbors的对错
            vicinity_correctness = (vicinity_predictions1_tmp == vicinity_gt_tmp).long()

            ## 分队的样本index
            batch_correct_index = (first_predicts1 == gt).long().nonzero(as_tuple=True)[0]
            batch_incorrect_index = (first_predicts1 != gt).long().nonzero(as_tuple=True)[0]
            ## 分错的样本


            # ### add this batch to the fron of the pool ##########
            index = index_tmp
            vicinity1 = vicinity1_tmp
            vicinity2 = vicinity2_tmp
            vicinity_predictions1 = vicinity_predictions1_tmp
            vicinity_predictions2 = vicinity_predictions2_tmp
            vicinity_confidence1 = vicinity_confidence1_tmp
            vicinity_confidence2 = vicinity_confidence2_tmp
            vicinity_gt = vicinity_gt_tmp


            range_index = torch.tensor(range(len(softmax_out1)))
            p11 = softmax_out1[range_index, first_predicts1] 
            p12 = softmax_out1[range_index, first_predicts2]
            p21 = softmax_out2[range_index, first_predicts1]
            p22 = softmax_out2[range_index, first_predicts2]
            aug_d = torch.ones(len(p11)).to(device)

            # pair-wise
            mask_self = (idx.unsqueeze(1) == index.unsqueeze(0)).long()
            s11, _ = convert_label_to_overlapMatrix(softmax_out1, vicinity1, first_predicts1)
            s22, _ = convert_label_to_overlapMatrix(softmax_out2, vicinity2, first_predicts2)
            s21, _ = convert_label_to_overlapMatrix(softmax_out1, vicinity2, first_predicts1)
            s12, _ = convert_label_to_overlapMatrix(softmax_out2, vicinity1, first_predicts2)
            
            p11_M, _ = convert_label_to_confMatrix(softmax_out1, vicinity1, first_predicts1)
            p22_M, _ = convert_label_to_confMatrix(softmax_out2, vicinity2, first_predicts2)
            p21_M, _ = convert_label_to_confMatrix(softmax_out1, vicinity2, first_predicts1)
            p12_M, _ = convert_label_to_confMatrix(softmax_out2, vicinity1, first_predicts2)
            # 属于周围相本的mask
            labelM1 = (first_predicts1.unsqueeze(1) == vicinity_predictions1.unsqueeze(0)).long()
            labelM2 = (first_predicts2.unsqueeze(1) == vicinity_predictions2.unsqueeze(0)).long()
            labelM21 = (first_predicts1.unsqueeze(1) == vicinity_predictions2.unsqueeze(0)).long()
            labelM12 = (first_predicts2.unsqueeze(1) == vicinity_predictions1.unsqueeze(0)).long()

            # 周围样本的weight
            s11_gk = gaussian_kernal(1-s11, confidence1, labelM1, mask_self)
            s12_gk = gaussian_kernal(1-s12, confidence2, labelM12, mask_self)
            s21_gk = gaussian_kernal(1-s21, confidence1, labelM21, mask_self)
            s22_gk = gaussian_kernal(1-s22, confidence2, labelM2, mask_self)

            # vrm_pairs_self = torch.cat((s11.mul(1-s11), s11.mul(1-s21), s22.mul(1-s22), s22.mul(1-s12)), dim=-1)
            # vrm_pairs_peer = torch.cat((s12.mul(1-s12), s12.mul(1-s22), s21.mul(1-s21), s21.mul(1-s11)),dim=-1)
    
            vrm_pairs_self = torch.cat((s11_gk.mul(1-p21_M).mul(labelM1), s21_gk.mul(1-p11_M).mul(labelM21)), dim=-1)
            vrm_pairs_peer = torch.cat((s22_gk.mul(1-p12_M).mul(labelM2), s12_gk.mul(1-p22_M).mul(labelM12)), dim=-1)  

            sd11 += s11_gk.sum()
            sd12 += s12_gk.sum()
            sd21 += s21_gk.sum()
            sd22 += s22_gk.sum()

            vrm_pairs_1_tmp = vrm_pairs_self.sum(dim=-1).div(s11_gk.mul(labelM1).sum(-1) + s21_gk.mul(labelM21).sum(-1)).sum()
            vrm_pairs_2_tmp = vrm_pairs_peer.sum(dim=-1).div(s22_gk.mul(labelM2).sum(-1) + s12_gk.mul(labelM12).sum(-1)).sum()

            vrm_pairs_1 += vrm_pairs_1_tmp
            vrm_pairs_2 += vrm_pairs_2_tmp

            vrm_pairs_all += vrm_pairs_1_tmp + vrm_pairs_2_tmp

            # compar EI + VRE
            # print(s11_gk.shape, p11.shape, p21_M.shape, labelM1.shape, labelM21.shape)
            # assert(0)
            wj_score_sum += torch.mul(confidence1[same_pred], confidence2[same_pred]).sqrt().sum()
            EI_randomPair = s22_gk.mul(p22_M.mul(p12_M).sqrt()).mul(labelM2.mul(labelM12))
            EI_randomPair_sum += EI_randomPair.sum(dim=-1).div(s22_gk.mul(labelM2).sum(-1)).sum()
            
            # neighbors的weight
            vicinity_weights = s22_gk*labelM1

            for i in range(len(idx)):
                for j in range(5):
                    
                    ## neighbors的index
                    neighbors_idx = labelM1[i].long().nonzero(as_tuple=True)[0]
                    neighbor_weights = vicinity_weights[i][neighbors_idx]

                    weight_low = neighbor_weights.sort()[0][int((j/5)*len(neighbor_weights))]
                    weight_high = neighbor_weights.sort()[0][int((j/5 + 0.2)*len(neighbor_weights))-1]

                    if j == 9: weight_high = 1
                    ## 符合范围的neighbors的index
                    target_neibors_index = torch.logical_and(neighbor_weights>weight_low, neighbor_weights<weight_high).long().nonzero(as_tuple=True)[0]
                    if i in batch_correct_index:
                        correct_weighVSacc_correctNum[j] += vicinity_correctness[neighbors_idx][target_neibors_index].sum()
                        correct_weighVSacc_totalNum[j] += len(target_neibors_index)
                    elif i in batch_incorrect_index:
                        incorrect_weighVSacc_correctNum[j] += vicinity_correctness[neighbors_idx][target_neibors_index].sum()
                        incorrect_weighVSacc_totalNum[j] += len(target_neibors_index)
                    else:
                        assert("error: code bug")

        for i in range(5):
            correct_weighVSacc[i] = (correct_weighVSacc_correctNum[i]/correct_weighVSacc_totalNum[i]).item()
            incorrect_weighVSacc[i] = (incorrect_weighVSacc_correctNum[i]/incorrect_weighVSacc_totalNum[i]).item()

        EI_randomPair_sum = EI_randomPair_sum/data_num
        wj_score_sum = wj_score_sum/data_num
        print(wj_score_sum)
        print(EI_randomPair_sum)

    return correct_weighVSacc, incorrect_weighVSacc

def main(args):
    setup_seed(0)

    model_name = args.model_name
    job_id = args.job_id
    print(model_name)
    
    
    if model_name == 'resnet50':
        model2_name = 'resnet50d'
    else:
        model2_name = 'resnet50'


    grey_prediction_path1 = '/root/paddlejob/workspace/env_run/afs/liuyuchi/autoEval/modelOutput/imagenet_r_out_rotation/%s.npy'%(model_name)
    grey_prediction_path2 = '/root/paddlejob/workspace/env_run/afs/liuyuchi/autoEval/modelOutput/imagenet_r_out_rotation/%s.npy'%(model2_name)

                      
    wj_test_dataset = ImageNetPredictions_gt(args.ImageNetV2_path, grey_prediction_path1, grey_prediction_path2, type='grey')      
                                   

    wj_test_loader = torch.utils.data.DataLoader(wj_test_dataset, batch_size=args.batch_size_fer,
                                              shuffle=True,
                                              num_workers=args.workers,
                                              pin_memory=True,
                                              drop_last=False)
    
    log_dir = os.path.join("/root/paddlejob/workspace/env_run/output/", "imagenet_r_out_rotation_neighorsVSacc")
    args.use_raw_imgs = False
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)

    # pool_manager = my_pool_manager(args, ganimation_model, erase_loader)
    # pool_manager.start()

    # if os.path.exists(os.path.join(log_dir, 'log.txt')):
    #     os.remove(os.path.join(log_dir, 'log.txt')) 
    device = torch.device('cuda:0')
    start_time = time.time()

    DoC_source_path = '/root/paddlejob/workspace/env_run/afs/liuyuchi/autoEval/DoC/imagenet_val_AC/%s.npy'%(model_name)
    ATC_source_path = '/root/paddlejob/workspace/env_run/afs/liuyuchi/autoEval/ATC/imagenet_val_ATC/%s.npy'%(model_name)
    S_AC, S_accuracy = Get_DoC_source(DoC_source_path)
    ATC_t = Get_ATC_source(ATC_source_path)
    
    exter_knowledge_dic = {}
    exter_knowledge_dic['S_AC']=S_AC
    exter_knowledge_dic['S_accuracy']=S_accuracy
    exter_knowledge_dic['ATC_t']=ATC_t


    correct_weighVSacc, incorrect_weighVSacc = test_wj(wj_test_loader, exter_knowledge_dic, device)
    print("--- %s seconds ---" % (time.time() - start_time))

    dic = {}
    dic['correct_weighVSacc'] = correct_weighVSacc
    dic['incorrect_weighVSacc'] = incorrect_weighVSacc

    print(correct_weighVSacc, '\n',incorrect_weighVSacc)


    np.save(os.path.join(log_dir ,model_name+'.npy'), dic)

  
    

opt = Options().parse()
main(opt)