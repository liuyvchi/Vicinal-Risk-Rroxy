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
from loss import ACLoss, convert_label_to_AUsim, AU_Loss, convert_label_to_EI, convert_label_to_overlap, convert_label_to_overlapMatrix, compute_vicinalRisk_L2, test_tprATfpr, roc_auc, convert_label_to_intensDis, mixup_data, mixup_criterion
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

def load_checkpoints(config_path, checkpoint_path, cpu=False):

    with open(config_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    if opt.kp_num != -1:
        config['model_params']['common_params']['num_kp'] = opt.kp_num
    generator = getattr(GEN, opt.generator)(**config['model_params']['generator_params'],**config['model_params']['common_params'])
    if not cpu:
        generator.cuda()
    config['model_params']['common_params']['num_channels'] = 4
    kp_detector = KPDetector(**config['model_params']['kp_detector_params'],
                             **config['model_params']['common_params'])
    if not cpu:
        kp_detector.cuda()
    
    if cpu:
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    else:
        checkpoint = torch.load(checkpoint_path,map_location="cuda:0")
    
    ckp_generator = OrderedDict((k.replace('module.',''),v) for k,v in checkpoint['generator'].items())
    generator.load_state_dict(ckp_generator)
    ckp_kp_detector = OrderedDict((k.replace('module.',''),v) for k,v in checkpoint['kp_detector'].items())
    kp_detector.load_state_dict(ckp_kp_detector)
    
    if not cpu:
        generator = DataParallelWithCallback(generator)
        kp_detector = DataParallelWithCallback(kp_detector)

    generator.eval()
    kp_detector.eval()
    
    return generator, kp_detector


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



def test(test_loader, device):
    predicted = []
    gt = []
    with torch.no_grad():

        running_loss = 0.0
        iter_cnt = 0
        correct_sum = 0
        au_intensT = [0, 0.4, 0.7, 1]
        correct_sum_small = [0 for i in range(3)]
        small_index = [[] for i in range(3)]
        small_num = [0 for i in range(3)]
        test_acc_small = [0 for i in range(3)]
        data_num = 0

        invariance_sum = 0
        aus_invariance_sum = 0
        sp_sum = 0
        aus_base_sum = 0
        integral_sum = 0
        integral_sum_2 = 0
        p_overlap_sum = 0
        pn_overlap_sum = 0
        self_vrm_sum =0

        aus_overlap_sum = 0
        aus_vrm_sum = 0
        aus_PNoverlap_sum = 0
        ensemble_overlap_sum = 0
        ensemble_Poverlap_sum = 0
        ensemble_PNoverlap_sum = 0
        ensemble_vrm_sum = 0
        mixup_overlap_sum = 0
        mixup_PNoverlap_sum = 0
        mixup_vrm_sum = 0
        ensemble_P_halfvrm_sum =0
        mixup_halfvrm_sum = 0

        ensemble_s1s2 = 0
        ensemble_Ps1s2 = 0
        # ensemble_Agreement = 0
        
        seed = 10
        torch.manual_seed(seed)
        for batch_i, (prediction1_1, prediction1_2, prediction2_1, prediction2_2, lamd, mix_idx, acc, gt) in enumerate(test_loader):

            output1_1 = prediction1_1.to(device)
            output1_2 = prediction1_2.to(device)
            output2_1 = prediction2_1.to(device)
            output2_2 = prediction2_2.to(device)
            lamd = lamd.to(device)
  
            raw_output = output1_1
            raw_softmaxOut = F.softmax(raw_output, dim=-1)
            _, raw_predicts = torch.topk(raw_output, k=2, dim=1)
            

            raw_output2 =output2_1
            raw_softmaxOut2 = F.softmax(raw_output2, dim=-1)
            _, ensemble_predicts = torch.topk(raw_output2, k=2, dim=1)
            first_predicts = ensemble_predicts[:, 0]

            p_overlap, n_overlap = convert_label_to_overlap(raw_softmaxOut, first_predicts)
            p_integral2, n_integral2 = convert_label_to_overlap(raw_softmaxOut2, first_predicts)

            v_index =  p_integral2>0.6
            nv_index = (n_integral2>0.1).logical_and(n_integral2<0.3)
            p_overlap, n_overlap = p_overlap[v_index], n_overlap[nv_index]
            p_integral2, n_integral2 = p_integral2[v_index], n_integral2[nv_index]

            ensemble_P_halfvrm_sum+= (1-p_integral2*(1-p_overlap)).mean()
            ensemble_Poverlap_sum += (2*p_integral2.mul(p_overlap) - p_integral2 - p_overlap + 1).mean()
            ensemble_overlap_sum += (2*p_integral2.mul(p_overlap) - p_integral2 - p_overlap + 1).mean()/2 + (2*n_integral2.mul(n_overlap) - n_integral2 - n_overlap + 1).mean()/2
            ensemble_vrm_sum += compute_vicinalRisk_L2(p_integral2, p_overlap)/2 + compute_vicinalRisk_L2(n_integral2, n_overlap)/2

            ensemble_s1s2 += p_integral2.mul(p_overlap).mean()/2 + n_integral2.mul(n_overlap).mean()/2
            ensemble_Ps1s2 += p_integral2.mul(p_overlap).mean()
            # change face ID for expressions
            fake_output, lamd, mix_idx = output1_2, lamd, mix_idx
            p_mix_idx = lamd>0.5

            fake_softmax = F.softmax(fake_output, dim=-1)
            fake_confidence, fake_predicts = fake_softmax.max(dim=-1)
            
            fake_overlap = torch.mul(raw_softmaxOut, fake_softmax).sum(dim=-1)

            lamd, fake_overlap = lamd[p_mix_idx], fake_overlap[p_mix_idx]

            mixup_halfvrm_sum += (1-lamd*(1-fake_overlap)).mean()
            mixup_vrm_sum += (2*lamd*fake_overlap - lamd - fake_overlap +1).mean()
            


            data_num+=len(acc)
            iter_cnt += 1

    return acc[0].item(), ensemble_s1s2.item()/iter_cnt, ensemble_Ps1s2.item()/iter_cnt, ensemble_P_halfvrm_sum.item()/iter_cnt, ensemble_Poverlap_sum.item()/iter_cnt, ensemble_overlap_sum.item()/iter_cnt, ensemble_vrm_sum.item()/iter_cnt, mixup_halfvrm_sum.item()/iter_cnt, mixup_vrm_sum.item()/iter_cnt

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

        seed = 10
        torch.manual_seed(seed)
        for batch_i, (prediction1_1, prediction1_2, prediction2_1, prediction2_2, acc, gt) in enumerate(test_loader):
            
            

            output1_1 = prediction1_1.to(device)
            output1_2 = prediction1_2.to(device)
            output2_1 = prediction2_1.to(device)
            output2_2 = prediction2_2.to(device)
            softmax_out1 =  F.softmax(output1_1, dim=-1)
            softmax_out2 = F.softmax(output1_2, dim=-1)
            confidence1 = softmax_out1.max(dim=-1)[0]
            confidence2 = softmax_out2.max(dim=-1)[0]

            ### filter the samples with low confidence
            # idx_lowConf = (confidence1+confidence2)<1
            # output1_1 = output1_1[idx_lowConf]
            # output1_2 = output1_2[idx_lowConf]
            # output2_1 = output2_1[idx_lowConf]
            # output2_2 = output2_2[idx_lowConf]
            # softmax_out1 = softmax_out1[idx_lowConf]
            # softmax_out2 = softmax_out2[idx_lowConf]
            # confidence1 = confidence1[idx_lowConf]
            # confidence2 = confidence2[idx_lowConf]


            _, predicts1 = torch.topk(output1_1, k=2, dim=1)
            _, predicts2 = torch.topk(output1_2, k=2, dim=1)
            first_predicts1 = predicts1[:, 0]
            first_predicts2 = predicts2[:, 0]
            same_pred = torch.eq(first_predicts1, first_predicts2)
            notsame_pred = ~same_pred
            wj_score_sum += torch.mul(confidence1[same_pred], confidence2[same_pred]).sqrt().sum()

            #### select samples with different predictions ##########
            # softmax_out1 = softmax_out1[notsame_pred]            
            # softmax_out2 = softmax_out2[notsame_pred]  
            # first_predicts1 = first_predicts1[notsame_pred]          
            # first_predicts2 = first_predicts2[notsame_pred]    
            # output2_1 = output2_1[notsame_pred]
            # output2_2 = output2_2[notsame_pred]      
            
            # p_overlap, n_overlap = convert_label_to_overlap(softmax_out1, first_predicts2)
            # p_integral2, n_integral2 = convert_label_to_overlap(softmax_out2, first_predicts2)
            # v_index =  p_overlap>0.5
            # nv_index = (n_overlap>0.3)
            # p_overlap, n_overlap = p_overlap[v_index], n_overlap[nv_index]
            # p_integral2, n_integral2 = p_integral2[v_index], n_integral2[nv_index]
            # s12 = torch.cat((p_overlap, n_overlap),dim=0)
            # s22 = torch.cat((p_integral2, n_integral2),dim=0)
 
            # p_overlap, n_overlap = convert_label_to_overlap(softmax_out1, first_predicts1)
            # p_integral2, n_integral2 = convert_label_to_overlap(softmax_out2, first_predicts1)
            # v_index =  p_integral2>0.5
            # nv_index = (n_integral2>0.3)
            # p_overlap, n_overlap = p_overlap[v_index], n_overlap[nv_index]
            # p_integral2, n_integral2 = p_integral2[v_index], n_integral2[nv_index]
            # s11 = torch.cat((p_overlap, n_overlap),dim=0)
            # s21 = torch.cat((p_integral2, n_integral2),dim=0)

            s11, labelM1 = convert_label_to_overlapMatrix(softmax_out1, softmax_out1, first_predicts1)
            s22, labelM2 = convert_label_to_overlapMatrix(softmax_out2, softmax_out2, first_predicts2)
            label_local12 = (first_predicts1.unsqueeze(1) == first_predicts2.unsqueeze(0)).long()
            label_local21 = (first_predicts2.unsqueeze(1) == first_predicts1.unsqueeze(0)).long()

            s12, _ = convert_label_to_overlapMatrix(softmax_out1, softmax_out2, first_predicts1)
            s21, _ = convert_label_to_overlapMatrix(softmax_out2, softmax_out1, first_predicts2)

            s12, s21 = s12.clamp(1e-8), s21.clamp_(1e-8)

            # vrm_pairs_self = torch.cat((s11.mul(1-s11), s11.mul(1-s21), s22.mul(1-s22), s22.mul(1-s12)), dim=-1)
            # vrm_pairs_peer = torch.cat((s12.mul(1-s12), s12.mul(1-s22), s21.mul(1-s21), s21.mul(1-s11)),dim=-1)
            vrm_pairs_self = torch.cat((s11.mul(1-s22).mul(labelM1), s22.mul(1-s11).mul(labelM2)), dim=-1)
            vrm_pairs_peer = torch.cat((s12.mul(1-s21).mul(label_local12), s21.mul(1-s12).mul(label_local21)), dim=-1) 

            sd11 += s11.sum()
            sd12 += s12.sum()
            sd21 += s21.sum()
            sd22 += s22.sum()

            vrm_pairs_1_tmp = vrm_pairs_self.sum(dim=-1).div(s11.mul(labelM1).sum(-1) + s22.mul(labelM2).sum(-1)).sum()
            vrm_pairs_2_tmp = vrm_pairs_peer.sum(dim=-1).div(s12.mul(label_local12).sum(-1).clamp_(1e-8) + s21.mul(label_local21).sum(-1).clamp_(1e-8)).sum()

            vrm_pairs_1 += vrm_pairs_1_tmp
            vrm_pairs_2 += vrm_pairs_2_tmp

            vrm_pairs_all += vrm_pairs_1_tmp + vrm_pairs_2_tmp

            range_index = torch.tensor(range(len(softmax_out1)))
            p11 = softmax_out1[range_index, first_predicts1] 
            p12 = softmax_out1[range_index, first_predicts2]
            p21 = softmax_out2[range_index, first_predicts1]
            p22 = softmax_out2[range_index, first_predicts2]
            aug_d = torch.ones(len(p11)).to(device)


            # EI_vrm_self = torch.cat((p11.mul(1-p11), p11.mul(1-p21), p22.mul(1-p22), p22.mul(1-p12)), dim=-1)
            EI_vrm_self = torch.cat((p11.mul(1-p21), p22.sqrt().mul(1-p12)), dim=-1)
            # EI_vrm_self = p11.mul(1-p21).div(p11+p22)+p22.mul(1-p12).div(p11+p22)
            # EI_vrm_self = torch.cat((aug_d.mul(1-p21), aug_d.mul(1-p12)), dim=-1)

            # EI_vrm_peer = torch.cat((p12.mul(1-p12), p12.mul(1-p22), p21.mul(1-p21), p21.mul(1-p11)),dim=-1)
            EI_vrm_peer = torch.cat((p12.mul(1-p22), p21.mul(1-p11)),dim=-1)
            # EI_vrm_peer = p12.mul(1-p22).div(p12+p21)+p21.mul(1-p11).div(p12+p21)
            # EI_vrm_peer = torch.cat((aug_d.mul(1-p22), aug_d.mul(1-p11)),dim=-1)


            EI_vrm_1 += EI_vrm_self.sum()
            EI_vrm_2 += EI_vrm_peer.sum()

            d11 += p11.sqrt().sum()
            d12 += p12.sqrt().sum()
            d21 += p21.sqrt().sum()
            d22 += p22.sqrt().sum()
            # d11 += aug_d.sum()
            # d12 += aug_d.sum()
            # d21 += aug_d.sum()
            # d22 += aug_d.sum()

            EI_vrm_all += EI_vrm_self.sum() + EI_vrm_peer.sum()
            # norm = p11+p22+p12+p21
            # EI_vrm_all += (p11.mul(1-p21).div(norm)+p22.mul(1-p12).div(norm)+p11.mul(1-p11).div(norm)+p22.mul(1-p22).div(norm)).sum()

            # EI_vrm = (2*p1on1.mul(p2on1) - p1on1 - p2on1 + 1).mean()/2 + (2*p1on2.mul(p2on2) - p1on2 - p2on2 + 1).mean()/2
            
            p_EI, n_EI = convert_label_to_EI(softmax_out1, first_predicts1)
            EI_randomPair_sum += p_EI.sqrt().sum()/(len(p_EI)+len(n_EI))

            output1_m2 = output2_1
            output2_m2 = output2_2
            _, predicts2_1 = torch.topk(output2_1, k=2, dim=1)
            _, predicts2_2 = torch.topk(output2_2, k=2, dim=1)
            first_predicts2_1 = predicts2_1[:, 0]
            first_predicts2_2 = predicts2_2[:, 0]
            same_pred_2m = torch.eq(first_predicts1, first_predicts2_1)
            same_pred_m2 = torch.eq(first_predicts2_1, first_predicts2_2)
            ensemble_agreement += same_pred_2m.sum()/len(output1_1)
            softmax_out1m2 = F.softmax(output1_m2, dim=-1)
            softmax_out2m2 = F.softmax(output2_m2, dim=-1)
            p_integral2 = torch.mul(softmax_out1m2, softmax_out2m2).sum(dim=-1)
            p_overlap = torch.mul(softmax_out1, softmax_out2).sum(dim=-1)
            grey_vrm_sum += 1 - p_integral2.mul(1 - p_overlap).mean()
            grey_vrmV2_sum += 1 - (1 - p_overlap).mean()
            grey_vrmV3_sum += 1 - p_integral2[same_pred_m2].mul(1 - p_overlap[same_pred_m2]).mean()
            
            iter_cnt += 1
            data_num+=len(softmax_out1)

        wj_score_sum = wj_score_sum/data_num
        EI_vrm_1 = 1 - EI_vrm_1/((d11+d22))
        EI_vrm_2 = 1 - EI_vrm_2/((d12+d21))
        EI_vrm_all =  1 - EI_vrm_all/((d11+d12+d21+d22).sum())
        vrm_pairs_1 = 1 - vrm_pairs_1/data_num
        vrm_pairs_all =  1 - vrm_pairs_all/(2*data_num)

    return acc[0].item(), wj_score_sum.item(), EI_randomPair_sum.item()/iter_cnt, grey_vrm_sum.item()/iter_cnt, grey_vrmV2_sum.item()/iter_cnt, grey_vrmV3_sum.item()/iter_cnt, EI_vrm_1.item(), EI_vrm_2.item(), EI_vrm_all.item(), vrm_pairs_1.item(), vrm_pairs_all.item(), ensemble_agreement.item()/iter_cnt


def main(args):
    setup_seed(0)

    model_name = args.model_name
    job_id = args.job_id
    print(model_name)
    
    
    if model_name == 'resnet50':
        model2_name = 'resnet50d'
    else:
        model2_name = 'resnet50'
    mixup_prediction_path1 = '/root/paddlejob/workspace/env_run/afs/liuyuchi/autoEval/modelOutput/imagenet_a_out_mixup/%s.npy'%(model_name)
    mixup_prediction_path2 = '/root/paddlejob/workspace/env_run/afs/liuyuchi/autoEval/modelOutput/imagenet_a_out_mixup/%s.npy'%(model2_name)

    grey_prediction_path1 = '/root/paddlejob/workspace/env_run/afs/liuyuchi/autoEval/modelOutput/imagenet_a_out_rotation/%s.npy'%(model_name)
    grey_prediction_path2 = '/root/paddlejob/workspace/env_run/afs/liuyuchi/autoEval/modelOutput/imagenet_a_out_rotation/%s.npy'%(model2_name)

    test_dataset = ImageNetPredictions(args.ImageNetV2_path, mixup_prediction_path1, mixup_prediction_path2, type='mixup')                         
    wj_test_dataset = ImageNetPredictions(args.ImageNetV2_path, grey_prediction_path1, grey_prediction_path2, type='grey')      
                                   
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size_fer,
                                              shuffle=True,
                                              num_workers=args.workers,
                                              pin_memory=True)
    wj_test_loader = torch.utils.data.DataLoader(wj_test_dataset, batch_size=args.batch_size_fer,
                                              shuffle=True,
                                              num_workers=args.workers,
                                              pin_memory=True)
    
    log_dir = os.path.join("/root/paddlejob/workspace/env_run/output/", "imagenet_a_rotation_V2")
    args.use_raw_imgs = False
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)

    # pool_manager = my_pool_manager(args, ganimation_model, erase_loader)
    # pool_manager.start()

    # if os.path.exists(os.path.join(log_dir, 'log.txt')):
    #     os.remove(os.path.join(log_dir, 'log.txt')) 
    device = torch.device('cuda:0')
    start_time = time.time()
    test_acc, ensemble_s1s2, ensemble_Ps1s2, ensemble_P_halfvrm, ensemble_Poverlap, ensemble_overlap, ensemble_vrm, mixup_halfvrm, mixup_vrm = test(test_loader, device)
    _, wj_score, EI_randomPair, grey_vrm, grey_vrmV2, grey_vrmV3, EI_vrm_1, EI_vrm_2, EI_vrm_all, vrm_pairs_1, vrm_pairs_all, ensemble_agreement = test_wj(wj_test_loader, device)
    print("--- %s seconds ---" % (time.time() - start_time))

    # print(i, train_acc, train_loss)
    dic = {}
    dic['test_acc'] = test_acc
    # dic['test_loss'] = test_loss
    # dic['confidence'] = confidence_mean
    dic['ensemble_P_halfvrm'] = ensemble_P_halfvrm
    dic['ensemble_Poverlap'] = ensemble_Poverlap
    dic['ensemble_overlap'] = ensemble_overlap
    dic['ensemble_vrm'] = ensemble_vrm

    dic['mixup_halfvrm'] = mixup_halfvrm
    dic['mixup_vrm'] = mixup_vrm

    dic['wj_score'] = wj_score

    dic['EI_randomPair'] = EI_randomPair
    dic['grey_vrm'] = grey_vrm
    dic['grey_vrmV2'] = grey_vrmV2
    dic['grey_vrmV3'] = grey_vrmV3

    dic['ensemble_s1s2'] = ensemble_s1s2
    dic['ensemble_Ps1s2'] = ensemble_Ps1s2
    dic['ensemble_agreement'] = ensemble_agreement
    dic['EI_vrm_1'] = EI_vrm_1
    dic['vrm_pairs_1'] = vrm_pairs_1
    dic['EI_vrm_all'] = EI_vrm_all
    dic['vrm_pairs_all'] = vrm_pairs_all

    print(test_acc, wj_score, ensemble_s1s2, ensemble_Ps1s2, ensemble_P_halfvrm, ensemble_Poverlap, ensemble_overlap, ensemble_vrm, mixup_halfvrm, mixup_vrm, EI_randomPair, grey_vrm, grey_vrmV2, grey_vrmV3, EI_vrm_1, EI_vrm_all, vrm_pairs_1, vrm_pairs_all, ensemble_agreement)
    np.save(os.path.join(log_dir ,model_name+'.npy'), dic)
    

opt = Options().parse()
main(opt)