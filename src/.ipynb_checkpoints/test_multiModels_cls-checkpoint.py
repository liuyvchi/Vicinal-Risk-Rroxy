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
from loss import ACLoss, convert_label_to_AUsim, AU_Loss, convert_label_to_EI, convert_label_to_overlap, compute_vicinalRisk_L2, test_tprATfpr, roc_auc, convert_label_to_intensDis, mixup_data, mixup_criterion
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
        
        seed = 10
        torch.manual_seed(seed)
        for batch_i, (prediction1_1, prediction1_2, prediction2_1, prediction2_2, lamd, mix_idx, acc) in enumerate(test_loader):

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

    return acc[0].item(), ensemble_P_halfvrm_sum.item()/iter_cnt, ensemble_Poverlap_sum.item()/iter_cnt, ensemble_overlap_sum.item()/iter_cnt, ensemble_vrm_sum.item()/iter_cnt, mixup_halfvrm_sum.item()/iter_cnt, mixup_vrm_sum.item()/iter_cnt

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
        grey_vrm_sum = 0
        rota_vrm_sum = 0

        seed = 10
        torch.manual_seed(seed)
        for batch_i, (prediction1_1, prediction1_2, prediction2_1, prediction2_2, acc) in enumerate(test_loader):

            output1_1 = prediction1_1.to(device)
            output1_2 = prediction1_2.to(device)
            output2_1 = prediction2_1.to(device)
            output2_2 = prediction2_2.to(device)

            _, predicts1 = torch.topk(output1_1, k=2, dim=1)
            _, predicts2 = torch.topk(output1_2, k=2, dim=1)
            first_predicts1 = predicts1[:, 0]
            first_predicts2 = predicts2[:, 0]
            same_pred = torch.eq(first_predicts1, first_predicts2)
            notsame_pred = ~same_pred
            softmax_out1 =  F.softmax(output1_1, dim=-1)
            softmax_out2 = F.softmax(output1_2, dim=-1)
            confidence1 = softmax_out1.max(dim=-1)[0]
            confidence2 = softmax_out2.max(dim=-1)[0]
            wj_score_sum += torch.mul(confidence1[same_pred], confidence2[same_pred]).sqrt().sum()/(len(output1_1))
            p_EI, n_EI = convert_label_to_EI(softmax_out1, first_predicts1)
            EI_randomPair_sum += p_EI.sqrt().sum()/(len(p_EI)+len(n_EI))

            output1_m2 = output2_1
            output2_m2 = output2_2
            softmax_out1m2 =  F.softmax(output1_m2, dim=-1)
            softmax_out2m2 = F.softmax(output2_m2, dim=-1)
            p_integral2 = torch.mul(softmax_out1m2, softmax_out2m2)
            p_overlap = torch.mul(softmax_out1, softmax_out2)
            grey_vrm_sum += 1 - (p_integral2.mul(1 - p_overlap)).mean()
            iter_cnt += 1

    return acc[0].item(), wj_score_sum.item()/iter_cnt, EI_randomPair_sum.item()/iter_cnt, grey_vrm_sum.item()/iter_cnt



def main(args):
    setup_seed(0)

    model_name = args.model_name
    job_id = args.job_id
    print(model_name)
    
    
    if model_name == 'swsl_resnext101_32x16d':
        model2_name = 'swsl_resnext101_32x8d'
    else:
        model2_name = 'swsl_resnext101_32x16d'
    mixup_prediction_path1 = '/root/paddlejob/workspace/env_run/afs/liuyuchi/autoEval/modelOutput/imagenet_r_out_mixup/%s.npy'%(model_name)
    mixup_prediction_path2 = '/root/paddlejob/workspace/env_run/afs/liuyuchi/autoEval/modelOutput/imagenet_r_out_mixup/%s.npy'%(model2_name)

    grey_prediction_path1 = '/root/paddlejob/workspace/env_run/afs/liuyuchi/autoEval/modelOutput/imagenet_r_out_grey/%s_imagenet_r.npy'%(model_name)
    grey_prediction_path2 = '/root/paddlejob/workspace/env_run/afs/liuyuchi/autoEval/modelOutput/imagenet_r_out_grey/%s_imagenet_r.npy'%(model2_name)

    test_dataset = ImageNetPredictions(mixup_prediction_path1, mixup_prediction_path2, type='mixup')                                        
    wj_test_dataset = ImageNetPredictions(grey_prediction_path1, grey_prediction_path2, type='grey')      
                                   
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size_fer,
                                              shuffle=True,
                                              num_workers=args.workers,
                                              pin_memory=True)
    wj_test_loader = torch.utils.data.DataLoader(wj_test_dataset, batch_size=args.batch_size_fer,
                                              shuffle=True,
                                              num_workers=args.workers,
                                              pin_memory=True)
    
    log_dir = os.path.join("/root/paddlejob/workspace/env_run/output/", "imagenet_r_vrm_ablation")
    args.use_raw_imgs = False
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)

    # pool_manager = my_pool_manager(args, ganimation_model, erase_loader)
    # pool_manager.start()

    # if os.path.exists(os.path.join(log_dir, 'log.txt')):
    #     os.remove(os.path.join(log_dir, 'log.txt')) 
    device = torch.device('cuda:0')
    start_time = time.time()
    test_acc, ensemble_P_halfvrm_mean, ensemble_Poverlap_mean, ensemble_overlap_mean, ensemble_vrm_mean, mixup_halfvrm_mean, mixup_vrm_mean = test(test_loader, device)
    _, wj_score, EI_randomPair, grey_vrm = test_wj(wj_test_loader, device)
    print("--- %s seconds ---" % (time.time() - start_time))

    # print(i, train_acc, train_loss)
    dic = {}
    dic['test_acc'] = test_acc
    # dic['test_loss'] = test_loss
    # dic['confidence'] = confidence_mean
    dic['ensemble_P_halfvrm_mean'] = ensemble_P_halfvrm_mean
    dic['ensemble_Poverlap_mean'] = ensemble_Poverlap_mean
    dic['ensemble_overlap'] = ensemble_overlap_mean
    dic['ensemble_vrm'] = ensemble_vrm_mean

    dic['mixup_halfvrm'] = mixup_halfvrm_mean
    dic['mixup_vrm'] = mixup_vrm_mean

    dic['wj_score'] = wj_score

    dic['EI_randomPair'] = EI_randomPair
    dic['grey_vrm'] = grey_vrm
    # dic['wjPlus_score'] = wjPlus_score_mean

    print(test_acc, wj_score, ensemble_P_halfvrm_mean, ensemble_Poverlap_mean, ensemble_overlap_mean, ensemble_vrm_mean, mixup_halfvrm_mean, mixup_vrm_mean, EI_randomPair, grey_vrm)
    np.save(os.path.join(log_dir ,model_name+'.npy'), dic)
    

opt = Options().parse()
main(opt)