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
from torchvision import transforms, datasets
import torchvision.models as models
import torch.utils.data as data
import torch.nn.functional as F
import timm

from dataset import RafDataset, MaEXDataset, AffectData, AffectData_twoTransforms, MS1MMaEX_Dataset, MS1MMaEXRec_Dataset, RafDataset_twoTransforms, ImageNetV, ImageNetV_twoTransforms, iwildcam_twoTransforms
from model import Model
from utils import *
# from resnet import *
from loss import ACLoss, convert_label_to_AUsim, AU_Loss, convert_label_to_EI, convert_label_to_overlap, compute_vicinalRisk_L2, test_tprATfpr, roc_auc, convert_label_to_intensDis, mixup_data, mixup_criterion
from backbones import get_model

from ImageNetAll import imagenet_a_mask, imagenet_r_mask, imagenet_o_mask
imagenet_mask = imagenet_o_mask



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

_torch_interpolation_to_str = {
    InterpolationMode.NEAREST: 'nearest',
    InterpolationMode.BILINEAR: 'bilinear',
    InterpolationMode.BICUBIC: 'bicubic',
    InterpolationMode.BOX: 'box',
    InterpolationMode.HAMMING: 'hamming',
    InterpolationMode.LANCZOS: 'lanczos',
}
_str_to_torch_interpolation = {b: a for a, b in _torch_interpolation_to_str.items()}


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

        seed = 10
        torch.manual_seed(seed)
        for batch_i, (imgs1, imgs2, raw_labels, index) in enumerate(test_loader):

            imgs1 = imgs1.to(device)
            raw_labels = raw_labels.to(device)

            # output1 = model(imgs1)[:,imagenet_mask]
            output1 = model(imgs1)
            iter_cnt += 1
            _, predicts = torch.max(output1, 1)

            # output2 = model(imgs2)[:,imagenet_mask]
            output2 = model(imgs2)

            correct_num = torch.eq(predicts, raw_labels).sum()
            correct_sum += correct_num
            data_num += len(raw_labels)

            for i in range(len(index)):
                if str(index[i]) not in results.keys():
                    results[str(index[i])] = [output1[i].cpu().numpy(), output2[i].cpu().numpy()]
                    
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
    best_or_last = 'best'

    # mixup_prediction_path1 = '/root/paddlejob/workspace/env_run/afs/liuyuchi/autoEval/modelOutput/imagenetv2_a_out_mixup/%s.npy'%(model_name)
    # grey_prediction_path1 = '/root/paddlejob/workspace/env_run/afs/liuyuchi/autoEval/modelOutput/imagenetv2_a_out_grey/%s.npy'%(model_name)
    rotation_prediction_path1 = '/root/paddlejob/workspace/env_run/afs/liuyuchi/autoEval/modelOutput/iwildcam_val_out_rotation/%s.npy'%(model_name+'_'+best_or_last)


    if not os.path.exists(rotation_prediction_path1):
        pass
    else:
        # print('predictions have already been saved in '+mixup_prediction_path1)
        print('predictions have already been saved in '+rotation_prediction_path1)
        assert(0)

    
    
    # define models
    constructor = getattr(torchvision.models, 'resnet50')
    model = constructor()
    #adjust the last layer
    last_layer_name = 'fc'
    d_features = getattr(model, last_layer_name).in_features
    d_out = 182
    last_layer = nn.Linear(d_features, d_out)
    setattr(model, last_layer_name, last_layer)
    model.d_out = d_out

    model = model.cuda(0)
    model = torch.nn.DataParallel(model, device_ids=args.gpu_ids)
    module_keys = model.module.state_dict().keys()
    
    # checkpoint = torch.load('/root/paddlejob/workspace/env_run/afs/liuyuchi/pretrain_weights_1000/%s.pth'%(model_name))
    ## iwildcaom
    model_string = best_or_last+'_model.pth'
    checkpoint_folder_path = '/root/paddlejob/workspace/env_run/Erasing-Attention-Consistency/iwildcam_weights/%s'%(model_name)
    files = os.listdir(checkpoint_folder_path)

    for file in files:
        if model_string in file:
            state = torch.load(os.path.join(checkpoint_folder_path, file))['algorithm']
            state = match_keys(state, list(module_keys))
            break

    model.module.load_state_dict(state, strict=False)

    ## the model remvoed the classifer
    device = torch.device('cuda:0')

    # # model2 = Model(args).cuda()
    # model2 = timm.create_model('resnet50', pretrained=False, num_classes=1000).cuda()
    # model2 = torch.nn.DataParallel(model2, device_ids=args.gpu_ids)
    # checkpoint = torch.load('/root/paddlejob/workspace/env_run/afs/liuyuchi/pretrain_weights_1000/resnet50.pth')
    # model2.module.load_state_dict(checkpoint, strict=True)
    
    # models for face change
    ganimation_model = create_model(args)

    #define datasets
    h = 448
    w = 448      
    
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    try:
        interpolation = _str_to_torch_interpolation[timm.data.resolve_data_config(model.module.pretrained_cfg)['interpolation']]
        print(timm.data.resolve_data_config(model.module.pretrained_cfg)['interpolation'])
    except:
        print('BILINEAR')
        interpolation = InterpolationMode.BILINEAR
    eval_transforms = transforms.Compose([transforms.ToPILImage(), transforms.Resize(w), transforms.CenterCrop((h,w)), transforms.ToTensor(), transforms.Normalize(mean, std)])
    # wjscore_transforms = transforms.Compose([transforms.ToPILImage(), transforms.Resize((h,w)), transforms.ToTensor(), transforms.Grayscale(num_output_channels=3), transforms.Normalize(mean, std)])
    wjscore_transforms = transforms.Compose([transforms.ToPILImage(), transforms.Resize(w), transforms.CenterCrop((h,w)), transforms.ToTensor(), MyRotationTransform(angles=[-90, 90, 180]), transforms.Normalize(mean, std)])
                                    
    wj_test_dataset = iwildcam_twoTransforms(args.ImageNetV2_path, transform=eval_transforms, transform2=wjscore_transforms)      
                                   

    wj_test_loader = torch.utils.data.DataLoader(wj_test_dataset, batch_size=args.batch_size_fer,
                                              shuffle=True,
                                              num_workers=args.workers,
                                              pin_memory=True)


    optimizer\
         = torch.optim.Adam(params=[{"params": model.parameters()}], lr=0.0001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)


    # test_acc, test_loss = test(model_ganimation, test_loader, device)
    # print(test_acc, test_loss)
    # assert (0)

    # log_dir = "log/R18MS1M_raf_AU_mixupHardAUs"
    
    log_dir = os.path.join("/root/paddlejob/workspace/env_run/output/", "iwildcam_idval_out_rotation")
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
    

    np.save(os.path.join(log_dir ,model_name+'_'+best_or_last+'.npy'), results_dic)

    

opt = Options().parse()
main(opt)