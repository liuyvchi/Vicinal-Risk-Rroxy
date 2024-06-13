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

from dataset import RafDataset, MaEXDataset, AffectData, AffectData_twoTransforms, MS1MMaEX_Dataset, \
    MS1MMaEXRec_Dataset, RafDataset_twoTransforms, ImageNetV, ImageNetV_twoTransforms
from model import Model
from utils import *
# from resnet import *
from loss import ACLoss, convert_label_to_AUsim, AU_Loss, convert_label_to_EI, convert_label_to_overlap, \
    compute_vicinalRisk_L2, test_tprATfpr, roc_auc, convert_label_to_intensDis, mixup_data, mixup_criterion

from ImageNetAll import imagenet_a_mask, imagenet_r_mask, imagenet_o_mask

imagenet_mask = imagenet_o_mask

# import torch.distributed as dist
# dist.init_process_group(backend='gloo|nccl')

## generation packages
from model_ganimation import create_model
from options import Options

import threading
import time


##DoGAN

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

            output1 = model(imgs1)[:,imagenet_mask]
            # output1 = model(imgs1)
            iter_cnt += 1
            _, predicts = torch.max(output1, 1)

            imgs2, targets_a, targets_b, lam, mix_index = mixup_data(imgs1, predicts, batch_i)

            output2 = model(imgs2)[:,imagenet_mask]
            # output2 = model(imgs2)

            correct_num = torch.eq(predicts, raw_labels).sum()
            correct_sum += correct_num
            data_num += len(raw_labels)

            for i in range(len(index)):
                if str(index[i]) not in results.keys():
                    results[str(index[i])] = [output1[i].cpu().numpy(), output2[i].cpu().numpy(), lam[i].cpu().numpy(),
                                              index[mix_index[i]]]

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

            output1 = model(imgs1)[:,imagenet_mask]
            # output1 = model(imgs1)
            iter_cnt += 1
            _, predicts = torch.max(output1, 1)

            output2 = model(imgs2)[:,imagenet_mask]
            # output2 = model(imgs2)

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

    # mixup_prediction_path1 = '/root/paddlejob/workspace/env_run/afs/liuyuchi/autoEval/modelOutput/imagenetv2_a_out_mixup/%s.npy'%(model_name)
    # grey_prediction_path1 = '/root/paddlejob/workspace/env_run/afs/liuyuchi/autoEval/modelOutput/imagenetv2_a_out_grey/%s.npy'%(model_name)
    rotation_prediction_path1 = './output/imagenet_val_out_rotation/%s.npy' % (model_name)

    if not os.path.exists(rotation_prediction_path1):
        pass
    else:
        # print('predictions have already been saved in '+mixup_prediction_path1)
        print('predictions have already been saved in ' + rotation_prediction_path1)
        assert (0)

    # define models

    model = timm.create_model(model_name, pretrained=True, num_classes=1000).cuda()
    model = torch.nn.DataParallel(model, device_ids=args.gpu_ids)

    ## the model remvoed the classifer
    model_feature = deepcopy(model)
    model_feature.module.reset_classifier(0)
    device = torch.device('cuda:0')

    # define datasets
    h = 224
    w = 224
    if '384' in model_name:
        h = 384
        w = 384
    if '256' in model_name:
        h = 256
        w = 256
    if '512' in model_name:
        h = 512
        w = 512
    if '448' in model_name:
        h = 448
        w = 448
        

    mean = model.module.pretrained_cfg['mean']
    std = model.module.pretrained_cfg['std']
    try:
        interpolation = _str_to_torch_interpolation[
            timm.data.resolve_data_config(model.module.pretrained_cfg)['interpolation']]
        print(timm.data.resolve_data_config(model.module.pretrained_cfg)['interpolation'])
    except:
        print('BILINEAR')
        interpolation = InterpolationMode.BILINEAR
    eval_transforms = transforms.Compose(
        [transforms.ToPILImage(), transforms.Resize(w), transforms.CenterCrop((h, w)), transforms.ToTensor(),
         transforms.Normalize(mean, std)])
    # wjscore_transforms = transforms.Compose([transforms.ToPILImage(), transforms.Resize((h,w)), transforms.ToTensor(), transforms.Grayscale(num_output_channels=3), transforms.Normalize(mean, std)])
    # wjscore_transforms = transforms.Compose([transforms.ToPILImage(), transforms.Resize(w), transforms.CenterCrop((h,w)), transforms.ToTensor(), MyRotationTransform(angles=[-90, 90, 180]), transforms.Normalize(mean, std)]) ## for objectnet
    # wjscore_transforms = transforms.Compose([transforms.ToPILImage(), transforms.Resize((h, w)), transforms.ToTensor(),
    #                                          transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2,
    #                                                                 hue=0.1)
    #                                             , transforms.Normalize(mean, std)])

    # wjscore_transforms = transforms.Compose([transforms.ToPILImage(), transforms.Resize((h, w)), transforms.ToTensor(),
    #                                          transforms.RandomErasing(p=1, scale=(0.1, 0.1), ratio=(0.3, 3.3), value=0, inplace=False)
    #                                             , transforms.Normalize(mean, std)])
    
    wjscore_transforms = transforms.Compose([transforms.ToPILImage(), transforms.Resize((h, w)), transforms.ToTensor(),
                                             transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)), transforms.Normalize(mean, std)])
    
    


    args.ImageNetV2_path = '../../data/objectnet/images'
    test_dataset = ImageNetV(args.ImageNetV2_path, transform=eval_transforms)
    wj_test_dataset = ImageNetV_twoTransforms(args.ImageNetV2_path, transform=eval_transforms,
                                              transform2=wjscore_transforms)

    args.batch_size_fer = 128
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size_fer,
                                              shuffle=True,
                                              num_workers=args.workers,
                                              pin_memory=True)
    wj_test_loader = torch.utils.data.DataLoader(wj_test_dataset, batch_size=args.batch_size_fer,
                                                 shuffle=True,
                                                 num_workers=args.workers,
                                                 pin_memory=True)

    optimizer \
        = torch.optim.Adam(params=[{"params": model.parameters()}], lr=0.0001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

    # test_acc, test_loss = test(model_ganimation, test_loader, device)
    # print(test_acc, test_loss)
    # assert (0)

    # log_dir = "log/R18MS1M_raf_AU_mixupHardAUs"

    log_dir = os.path.join("../modelOutput/", "objectnet_out_gaussianblur")
    args.use_raw_imgs = False
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)

    # pool_manager = my_pool_manager(args, ganimation_model, erase_loader)
    # pool_manager.start()

    # if os.path.exists(os.path.join(log_dir, 'log.txt')):ss
    #     os.remove(os.path.join(log_dir, 'log.txt'))

    start_time = time.time()
    results_dic = get_grey_predictions(model, wj_test_loader, device)
    print("--- %s seconds ---" % (time.time() - start_time))
    print(results_dic['acc'])

    np.save(os.path.join(log_dir, model_name + '.npy'), results_dic)


opt = Options().parse()
main(opt)