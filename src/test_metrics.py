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
from loss import ACLoss, convert_label_to_AUsim, AU_Loss, convert_label_to_EI, convert_label_to_overlap, convert_label_to_overlapMatrix, convert_label_to_cdistMatrix, gaussian_kernal, convert_label_to_confMatrix, compute_vicinalRisk_L2, test_tprATfpr, roc_auc, convert_label_to_intensDis, mixup_data, mixup_criterion

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

def test_wj(test_loader, exter_knowledge_dic, model_name, device):

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

        seed = 10
        torch.manual_seed(seed)
        for batch_i, (prediction1_1, prediction1_2, prediction2_1, prediction2_2, acc, idx) in enumerate(test_loader):
            output1_1 = prediction1_1.to(device)
            output1_2 = prediction1_2.to(device)
            softmax_out1 =  F.softmax(output1_1, dim=-1)
            softmax_out2 = F.softmax(output1_2, dim=-1)
            idx = idx.to(device)

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

        index_pool = torch.cat(index_tmp, dim=0)
        length = len(index_pool)
        # print(length)
        # assert(0)
        vicinity1_pool = torch.cat(vicinity1_tmp, dim=0)
        vicinity2_pool = torch.cat(vicinity2_tmp, dim=0)
        vicinity_predictions1_pool = torch.cat(vicinity_predictions1_tmp, dim=0)
        vicinity_predictions2_pool = torch.cat(vicinity_predictions2_tmp, dim=0)
        vicinity_confidence1_pool = torch.cat(vicinity_confidence1_tmp, dim=0)
        vicinity_confidence2_pool = torch.cat(vicinity_confidence2_tmp, dim=0)

        # similarity_pool = torch.zeros(7500, 7500).cuda()

        # laod_similarity = True
        # if laod_similarity == True:
        #     similarity_path = '/root/paddlejob/workspace/env_run/output/imagenet_a_Similarity_colorjitter/imagenet_a_Similarity_colorjitter_%s.npy'%(model_name)
        #     similarity_pool_load = torch.tensor(np.load(similarity_path)).cuda()
        #     similarity_pool_load = torch.tensor(np.load(similarity_path)).cuda()
        #     similarity_pool_load = similarity_pool_load[:, index_pool]


        for batch_i, (prediction1_1, prediction1_2, prediction2_1, prediction2_2, acc, idx) in enumerate(test_loader):
            
            ## random select neighbors
            # selected_num = 2000
            # ranom_start = random.randint(0, length-selected_num-1)
            # index_tmp = index_pool[ranom_start:ranom_start+selected_num]
            # vicinity1_tmp = vicinity1_pool[ranom_start:ranom_start+selected_num]
            # vicinity2_tmp = vicinity2_pool[ranom_start:ranom_start+selected_num]
            # vicinity_predictions1_tmp = vicinity_predictions1_pool[ranom_start:ranom_start+selected_num]
            # vicinity_predictions2_tmp = vicinity_predictions2_pool[ranom_start:ranom_start+selected_num]
            # vicinity_confidence1_tmp = vicinity_confidence1_pool[ranom_start:ranom_start+selected_num]
            # vicinity_confidence2_tmp = vicinity_confidence2_pool[ranom_start:ranom_start+selected_num]
            ##########

            output1_1 = prediction1_1.to(device)
            output1_2 = prediction1_2.to(device)
            # output2_1 = prediction2_1.to(device)
            # output2_2 = prediction2_2.to(device)
            softmax_out1 =  F.softmax(output1_1, dim=-1)
            softmax_out2 = F.softmax(output1_2, dim=-1)
            confidence1 = softmax_out1.max(dim=-1)[0]
            confidence2 = softmax_out2.max(dim=-1)[0]
            idx = idx.to(device)

            _, predicts1 = torch.topk(output1_1, k=2, dim=1)
            _, predicts2 = torch.topk(output1_2, k=2, dim=1)
            first_predicts1 = predicts1[:, 0]
            first_predicts2 = predicts2[:, 0]
            same_pred = torch.eq(first_predicts1, first_predicts2)
            notsame_pred = ~same_pred
            

            # ### add this batch to the fron of the pool ##########
            # index = torch.cat((idx, index_tmp),dim=0)
            # vicinity1 = torch.cat((softmax_out1, vicinity1_tmp), dim=0)
            # vicinity2 =torch.cat((softmax_out2, vicinity2_tmp), dim=0)
            # vicinity_predictions1 = torch.cat((first_predicts1, vicinity_predictions1_tmp), dim=0)
            # vicinity_predictions2 = torch.cat((first_predicts2, vicinity_predictions2_tmp), dim=0)
            # vicinity_confidence1 = torch.cat((confidence1, vicinity_confidence1_tmp), dim=0)
            # vicinity_confidence2 = torch.cat((confidence2, vicinity_confidence2_tmp), dim=0)
            index = index_pool
            vicinity1 = vicinity1_pool
            vicinity2 = vicinity2_pool
            vicinity_predictions1 = vicinity_predictions1_pool
            vicinity_predictions2 = vicinity_predictions2_pool
            vicinity_confidence1 = vicinity_confidence1_pool
            vicinity_confidence2 = vicinity_confidence2_pool


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

            # for i in range(len(idx)):
            #     similarity_pool[idx[i]][index] = s22[i]

            ## None transformation:
            
            # if laod_similarity == True:
            #     s22 = similarity_pool_load[idx]
          
            
            # equal similarity
            
            p11_M, _ = convert_label_to_confMatrix(softmax_out1, vicinity1, first_predicts1)
            p22_M, _ = convert_label_to_confMatrix(softmax_out2, vicinity2, first_predicts2)
            p21_M, _ = convert_label_to_confMatrix(softmax_out1, vicinity2, first_predicts1)
            p12_M, _ = convert_label_to_confMatrix(softmax_out2, vicinity1, first_predicts2)
            labelM1 = (first_predicts1.unsqueeze(1) == vicinity_predictions1.unsqueeze(0)).long()
            labelM2 = (first_predicts2.unsqueeze(1) == vicinity_predictions2.unsqueeze(0)).long()
            labelM21 = (first_predicts1.unsqueeze(1) == vicinity_predictions2.unsqueeze(0)).long()
            labelM12 = (first_predicts2.unsqueeze(1) == vicinity_predictions1.unsqueeze(0)).long()

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

            AC += confidence1.sum()
            AC_vrp = s22_gk.mul(vicinity_confidence1).mul(labelM2)
            AC_vrp_sum += AC_vrp.sum(dim=-1).div(s22_gk.mul(labelM2).sum(-1)).sum()

            ATC_acc += (confidence1>S_ATC_t).sum()
            ATC_vrp_acc = s22_gk.mul((vicinity_confidence1>S_ATC_t).long()).mul(labelM2)
            ATC_vrp_acc_sum +=  ATC_vrp_acc.sum(dim=-1).div(s22_gk.mul(labelM2).sum(-1)).sum()
            
            predict_Inv += same_pred.sum()
            predict_Inv_vrp = s22_gk.mul((vicinity_predictions1==vicinity_predictions2).long()).mul(labelM2)
            predict_Inv_vrp_sum += predict_Inv_vrp.sum(dim=-1).div(s22_gk.mul(labelM2).sum(-1)).sum()

            # difference of confidence of two views on the predicted class 
            confidence_Inv += (1-p21).sum()
            confidence_Inv_vrp = s22_gk.mul(1-p21_M).mul(labelM2)
            confidence_Inv_vrp_sum += confidence_Inv_vrp.sum(dim=-1).div(s22_gk.mul(labelM2).sum(-1)).sum()



            # EI_vrm_self = torch.cat((p11.mul(1-p11), p11.mul(1-p21), p22.mul(1-p22), p22.mul(1-p12)), dim=-1)
            EI_vrm_self = (p11.mul(1-p21)+p21.mul(1-p11)).div(p11+p21)

    
            # EI_vrm_self = p11.mul(1-p21).div(p11+p22)+p22.mul(1-p12).div(p11+p22)
            # EI_vrm_self = torch.cat((aug_d.mul(1-p21), aug_d.mul(1-p12)), dim=-1)

            # EI_vrm_peer = torch.cat((p12.mul(1-p12), p12.mul(1-p22), p21.mul(1-p21), p21.mul(1-p11)),dim=-1)
            EI_vrm_peer = (p22.mul(1-p12)+p12.mul(1-p22)).div(p22+p12)
            # EI_vrm_peer = p12.mul(1-p22).div(p12+p21)+p21.mul(1-p11).div(p12+p21)
            # EI_vrm_peer = torch.cat((aug_d.mul(1-p22), aug_d.mul(1-p11)),dim=-1)


            EI_vrm_1 += EI_vrm_self.sum()
            EI_vrm_2 += EI_vrm_peer.sum()

            d11 += p11.sum()
            d12 += p12.sum()
            d21 += p21.sum()
            d22 += p22.sum()
            # d11 += aug_d.sum()
            # d12 += aug_d.sum()
            # d21 += aug_d.sum()
            # d22 += aug_d.sum()

            EI_vrm_all += EI_vrm_self.sum() + EI_vrm_peer.sum()
            # norm = p11+p22+p12+p21
            # EI_vrm_all += (p11.mul(1-p21).div(norm)+p22.mul(1-p12).div(norm)+p11.mul(1-p11).div(norm)+p22.mul(1-p22).div(norm)).sum()

            # EI_vrm = (2*p1on1.mul(p2on1) - p1on1 - p2on1 + 1).mean()/2 + (2*p1on2.mul(p2on2) - p1on2 - p2on2 + 1).mean()/2

            
            iter_cnt += 1
            data_num+=len(softmax_out1)

        wj_score_sum = wj_score_sum/data_num
        EI_vrm_1 = 1 - EI_vrm_1/data_num
        EI_vrm_2 = 1 - EI_vrm_2/data_num
        EI_vrm_all =  1 - EI_vrm_all/(2*data_num)
        vrm_pairs_1 = 1 - vrm_pairs_1/data_num
        vrm_pairs_all =  1 - vrm_pairs_all/(2*data_num)
        EI_randomPair_sum = EI_randomPair_sum/data_num
        

        AC = AC/data_num
        AC_vrp_sum = AC_vrp_sum/data_num
        predict_Inv= predict_Inv/data_num
        predict_Inv_vrp_sum= predict_Inv_vrp_sum/data_num
        confidence_Inv = 1 - confidence_Inv/data_num
        confidence_Inv_vrp_sum = 1 - confidence_Inv_vrp_sum/data_num

        DoC_acc = S_accuracy - (S_AC - AC.item())
        DoC_vrp_acc = S_accuracy - (S_AC - AC_vrp_sum.item())

        ATC_acc = ATC_acc.item()/data_num
        ATC_vrp_acc_sum = ATC_vrp_acc_sum.item()/data_num

        # simi_save_dir = "/root/paddlejob/workspace/env_run/output/imagenet_a_Similarity_colorjitter"
        # if not os.path.exists(simi_save_dir):
        #     os.mkdir(simi_save_dir)
        # np.save(os.path.join(simi_save_dir, 'imagenet_a_Similarity_colorjitter_%s.npy'%(model_name)) , similarity_pool.cpu().numpy())


    return acc[0].item(),  DoC_acc, DoC_vrp_acc, ATC_acc, ATC_vrp_acc_sum, predict_Inv.item(), predict_Inv_vrp_sum.item(), confidence_Inv.item(), confidence_Inv_vrp_sum.item(), AC.item(), AC_vrp_sum.item(), wj_score_sum.item(), EI_randomPair_sum.item(), EI_vrm_1.item(), EI_vrm_2.item(), EI_vrm_all.item(), vrm_pairs_1.item(), vrm_pairs_all.item()


def main(args):
    setup_seed(0)

    model_name = args.model_name
    job_id = args.job_id
    print(model_name)
    
    
    if model_name == 'resnet50':
        model2_name = 'resnet50d'
    else:
        model2_name = 'resnet50'


    grey_prediction_path1 = '/home/liuyc/calibration/modelOutput/imagenet_a_out_rotation/%s.npy'%(model_name)
    grey_prediction_path2 = '/home/liuyc/calibration/modelOutput/imagenet_a_out_rotation/%s.npy'%(model2_name)

                      
    wj_test_dataset = ImageNetPredictions(args.ImageNetV2_path, grey_prediction_path1, grey_prediction_path2, type='grey', set_portion=args.set_portion)      
                                   

    wj_test_loader = torch.utils.data.DataLoader(wj_test_dataset, batch_size=args.batch_size_fer,
                                              shuffle=True,
                                              num_workers=args.workers,
                                              pin_memory=True,
                                              drop_last=False)
    
    log_dir = os.path.join("/root/paddlejob/workspace/env_run/output/", args.out_dir)
    args.use_raw_imgs = False
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)

    # pool_manager = my_pool_manager(args, ganimation_model, erase_loader)
    # pool_manager.start()

    # if os.path.exists(os.path.join(log_dir, 'log.txt')):
    #     os.remove(os.path.join(log_dir, 'log.txt')) 
    device = torch.device('cuda:0')
    start_time = time.time()

    DoC_source_path = '/home/liuyc/vicinal/output/DoC/imagenet_val_AC/%s.npy'%(model_name)
    ATC_source_path = '/home/liuyc/vicinal/output/ATC/imagenet_val_ATC/%s.npy'%(model_name)
    S_AC, S_accuracy = Get_DoC_source(DoC_source_path)
    ATC_t = Get_ATC_source(ATC_source_path)
    
    exter_knowledge_dic = {}
    exter_knowledge_dic['S_AC']=S_AC
    exter_knowledge_dic['S_accuracy']=S_accuracy
    exter_knowledge_dic['ATC_t']=ATC_t


    test_acc, DoC_acc, DoC_vrp_acc, ATC_acc, ATC_vrp_acc, predict_Inv, predict_Inv_vrp, confidence_Inv, confidence_Inv_vrp, AC, AC_vrp, wj_score, EI_randomPair, EI_vrm_1, EI_vrm_2, EI_vrm_all, vrm_pairs_1, vrm_pairs_all = test_wj(wj_test_loader, exter_knowledge_dic, model_name, device)
    print("--- %s seconds ---" % (time.time() - start_time))

    # print(i, train_acc, train_loss)
    dic = {}
    dic['test_acc'] = test_acc
    dic['AC'] = AC
    dic['AC_vrp'] = AC_vrp
    dic['predict_Inv'] = predict_Inv
    dic['predict_Inv_vrp'] = predict_Inv_vrp
    dic['confidence_Inv'] = confidence_Inv
    dic['confidence_Inv_vrp'] = confidence_Inv_vrp
    dic['DoC_acc'] = DoC_acc
    dic['DoC_vrp_acc'] = DoC_vrp_acc
    dic['ATC_acc'] = ATC_acc
    dic['ATC_vrp_acc'] = ATC_vrp_acc
    # dic['test_loss'] = test_loss
    # dic['confidence'] = confidence_mean

    dic['wj_score'] = wj_score

    dic['EI_randomPair'] = EI_randomPair


    dic['EI_vrm_1'] = EI_vrm_1
    dic['EI_vrm_2'] = EI_vrm_2
    dic['EI_vrm_all'] = EI_vrm_all
    dic['vrm_pairs_1'] = vrm_pairs_1
    dic['vrm_pairs_all'] = vrm_pairs_all

    print(test_acc, DoC_acc, DoC_vrp_acc, ATC_acc, ATC_vrp_acc, predict_Inv, predict_Inv_vrp, confidence_Inv, confidence_Inv_vrp, AC, AC_vrp, wj_score, EI_randomPair, EI_vrm_1, EI_vrm_2, EI_vrm_all, vrm_pairs_1, vrm_pairs_all)
    np.save(os.path.join(log_dir ,model_name+'.npy'), dic)
    

opt = Options().parse()
main(opt)