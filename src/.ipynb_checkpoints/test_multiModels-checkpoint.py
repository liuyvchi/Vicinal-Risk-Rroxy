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
from torchvision import transforms
import torchvision.models as models
import torch.utils.data as data
import torch.nn.functional as F
import timm

from dataset import RafDataset, MaEXDataset, AffectData, AffectData_twoTransforms, MS1MMaEX_Dataset, MS1MMaEXRec_Dataset, RafDataset_twoTransforms
from model import Model
from utils import *
# from resnet import *
from loss import ACLoss, convert_label_to_AUsim, AU_Loss, convert_label_to_overlap, compute_vicinalRisk_L2, test_tprATfpr, roc_auc, convert_label_to_intensDis, mixup_data, mixup_criterion
from backbones import get_model


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

def make_animation(source, driving_frame, generator, kp_detector, depth_encoder, depth_decoder, relative=False, adapt_movement_scale=True, cpu=False):
    

    with torch.no_grad():
        outputs = depth_decoder(depth_encoder(source))
        depth_source = outputs[("disp", 0)]
        source_kp = torch.cat((source, depth_source),1)
        kp_source = kp_detector(source_kp)

        outputs = depth_decoder(depth_encoder(driving_frame))
        depth_driving = outputs[("disp", 0)]
        driving_kp = torch.cat((driving_frame, depth_driving),1)
        kp_driving = kp_detector(driving_kp)

        # kp_norm = normalize_kp(kp_source=kp_source, kp_driving=kp_driving,
        #                            kp_driving_initial=kp_driving, use_relative_movement=False,
        #                            use_relative_jacobian=False, adapt_movement_scale=False)

        out = generator(source, kp_source=kp_source, kp_driving=kp_driving, source_depth = depth_source, driving_depth = depth_driving)

    return out['prediction']*255



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



def test(model, model2, model_feature, DaGan_gen, DaGan_detector, depth_encoder, depth_decoder, test_loader, eval_transforms, device):
    predicted = []
    gt = []
    with torch.no_grad():
        model.eval()
        model_feature.eval()
        model2.eval()

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
        ensemble_PNoverlap_sum = 0
        ensemble_vrm_sum = 0
        mixup_overlap_sum = 0
        mixup_PNoverlap_sum = 0
        mixup_vrm_sum = 0
        


        seed = 10
        torch.manual_seed(seed)
        for batch_i, (raw_imgs, ganimation_imgs, real_aus, aus_valid, raw_labels, indexes) in enumerate(test_loader):
            aus_valid[raw_labels==6] = False
            raw_imgs = raw_imgs.to(device)
            real_aus = real_aus.to(device)
            raw_labels = raw_labels.to(device)
            ganimation_imgs = ganimation_imgs.to(device)
            # DoGAN_imgs = DoGAN_imgs.to(device)

            raw_output = model(raw_imgs)
            raw_softmaxOut = F.softmax(raw_output, dim=-1)
            raw_confidence = raw_softmaxOut.max(-1)[0]
            _, raw_predicts = torch.topk(raw_output, k=2, dim=1)
            first_predicts = raw_predicts[:, 0]
            raw_correct_num = torch.eq(first_predicts, raw_labels).sum()
            wrong_index = torch.eq(first_predicts, raw_labels) == 0
            second_predicts = raw_predicts[:, 1]

            raw_output2 = model2(raw_imgs)
            raw_softmaxOut2 = F.softmax(raw_output2, dim=-1)
            p_integral2, n_integral2 = convert_label_to_overlap(raw_softmaxOut2, first_predicts)

            

            valid_aus = real_aus[aus_valid]
            valid_label = raw_labels[aus_valid]

            for i in range(len(correct_sum_small)):
                small_index[i] = deepcopy(aus_valid)
                small_index[i][(real_aus.max(dim=-1)[0] < au_intensT[i]).logical_and(real_aus.max(dim=-1)[0] > au_intensT[i+1])] = False

            # sp_aus, sn_aus = convert_label_to_AUsim(nn.functional.normalize(valid_aus), first_predicts[aus_valid])
            sp_aus, sn_aus = convert_label_to_AUsim(nn.functional.normalize(valid_aus), first_predicts[aus_valid])
            p_overlap, n_overlap = convert_label_to_overlap(raw_softmaxOut, first_predicts)
            p_overlap_v, n_overlap_v = convert_label_to_overlap(raw_softmaxOut[aus_valid], first_predicts[aus_valid])

            aus_overlap_sum += ((2*sp_aus.mul(p_overlap_v) - sp_aus - p_overlap_v + 1).sum() + (2*sn_aus.mul(n_overlap_v) - sn_aus - n_overlap_v + 1).sum())/(len(sp_aus)+len(sn_aus))
            aus_PNoverlap_sum += sp_aus.mul(p_overlap_v).mean() - sn_aus.mul(0.5 - n_overlap_v).mean()
            aus_vrm_sum += compute_vicinalRisk_L2(torch.cat((sp_aus, sn_aus), dim=0), torch.cat((p_overlap_v, n_overlap_v), dim=0))

            ensemble_overlap_sum += ((2*p_integral2.mul(p_overlap) - p_integral2 - p_overlap + 1).sum() + (2*n_integral2.mul(n_overlap) - n_integral2 - n_overlap + 1).sum())/(len(p_integral2)+len(n_integral2))
            ensemble_PNoverlap_sum += p_integral2.mul(p_overlap).mean() - n_integral2.mul(0.5 - n_overlap).mean()
            ensemble_vrm_sum += compute_vicinalRisk_L2(torch.cat((p_integral2, n_integral2), dim=0), torch.cat((p_overlap, n_overlap), dim=0))

            # change face ID for expressions
            fake_img, targets_a, targets_b, lam, index = mixup_data(raw_imgs, first_predicts, batch_i)

            # source_image = DoGAN_imgs.flip(dims=[0])
            # driving_video = DoGAN_imgs
            # DaGAN_fake = make_animation(source_image, driving_video, DaGan_gen, DaGan_detector, depth_encoder, depth_decoder)
            # # DaGAN_fake = DaGAN_fake.to(torch.uint8)

            # img_pil = transforms.ToPILImage()(DaGAN_fake.to(torch.uint8)[1])
            # img_pil.save('test_2.jpg')

            # print(DaGAN_fake[0,1,20])
            # assert(0)

            # fake_img = [eval_transforms(DaGAN_fake[i]) for i in range(len(DaGAN_fake))]
            # fake_softmax = F.softmax(fake_output, dim=-1)
            # fake_confidence, fake_predicts = fake_softmax.max(dim=-1)
            # same_pred = torch.eq(first_predicts[aus_valid], fake_predicts)
            # wjPlus_score_sum += torch.mul(raw_confidence[aus_valid][same_pred], fake_confidence[same_pred]).sqrt().sum()/(len(aus_valid))
            fake_output = model(fake_img)
            fake_softmax = F.softmax(fake_output, dim=-1)
            fake_confidence, fake_predicts = fake_softmax.max(dim=-1)
            same_pred_1= torch.eq(first_predicts, fake_predicts)
            same_pred_2 = torch.eq(first_predicts[index], fake_predicts)
            fake_overlap_1 = torch.mul(raw_softmaxOut, fake_softmax).sum(dim=-1)
            fake_overlap_2 = torch.mul(raw_softmaxOut[index], fake_softmax).sum(dim=-1)
            # mixup_loss = mixup_criterion(nn.CrossEntropyLoss(reduction='none'), fake_output, targets_a, targets_b, lam)
            mixup_overlap_sum += ((2*lam.mul(fake_overlap_1) - lam - fake_overlap_1 + 1).sum() + (2*(1-lam).mul(fake_overlap_2) - (1-lam) - fake_overlap_2 + 1).sum())/(2*len(lam))
            mixup_PNoverlap_sum +=  lam[same_pred_1].mul(fake_overlap_1[same_pred_1]).mean() - lam[~same_pred_1].mul(0.5 - fake_overlap_1[~same_pred_1]).mean()
            mixup_PNoverlap_sum +=  (1-lam)[same_pred_2].mul(fake_overlap_2[same_pred_2]).mean() - lam[~same_pred_2].mul(0.5 - fake_overlap_2[~same_pred_2]).mean()
            mixup_vrm_sum += compute_vicinalRisk_L2(torch.cat((lam, (1-lam)), dim=0), torch.cat((fake_overlap_1, fake_overlap_2,), dim=0))
           
            raw_feature = model_feature(raw_imgs)
            p_featDot, n_featDot = convert_label_to_overlap(nn.functional.normalize(raw_feature), first_predicts)
            self_vrm_sum += compute_vicinalRisk_L2(torch.cat((p_featDot, n_featDot), dim=0), torch.cat((p_overlap, n_overlap), dim=0))
            
            loss = nn.CrossEntropyLoss()(raw_output, raw_labels)

            iter_cnt += 1
            _, predicts = torch.max(raw_output, 1)

            predicted.append(predicts.cpu().detach())
            gt.append(raw_labels.cpu().detach())

            correct_sum += torch.eq(predicts, raw_labels).sum()
            for i in range(len(correct_sum_small)):
                correct_sum_small[i] += torch.eq(predicts[small_index[i]], raw_labels[small_index[i]]).sum()
                small_num[i] += len(predicts[small_index[i]])

            running_loss += loss
            data_num += raw_output.size(0)

        running_loss = running_loss / iter_cnt
        test_acc = correct_sum.float() / float(data_num)

        for i in range(len(correct_sum_small)):
            test_acc_small[i] = (correct_sum_small[i].float() / float(small_num[i])).item()

        # ## confusion matrix
        # predicted = torch.cat(predicted, dim=-1).numpy()
        # gt = torch.cat(gt, dim=-1).numpy()
        # print(metrics.classification_report(gt, predicted))

    return test_acc.item(), test_acc_small, aus_overlap_sum.item()/iter_cnt, aus_PNoverlap_sum.item()/iter_cnt, aus_vrm_sum.item()/iter_cnt, \
    ensemble_overlap_sum.item()/iter_cnt, ensemble_PNoverlap_sum.item()/iter_cnt, ensemble_vrm_sum.item()/iter_cnt, mixup_overlap_sum.item()/iter_cnt, \
    mixup_PNoverlap_sum.item()/iter_cnt, mixup_vrm_sum.item()/iter_cnt, self_vrm_sum.item()/iter_cnt

def test_wj(model, ganimation_model, test_loader, device):


    predicted = []
    gt = []
    with torch.no_grad():
        model.eval()

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

        seed = 10
        torch.manual_seed(seed)
        for batch_i, (imgs1, imgs2, real_aus, aus_valid, raw_labels, indexes) in enumerate(test_loader):

            aus_valid[raw_labels==6] = False
            imgs1 = imgs1.to(device)
            imgs2 = imgs2.to(device)
            real_aus = real_aus.to(device)
            raw_labels = raw_labels.to(device)
            valid_aus = real_aus[aus_valid]

            output1 = model(imgs1)
            output2 = model(imgs2)

            _, predicts1 = torch.topk(output1, k=2, dim=1)
            _, predicts2 = torch.topk(output2, k=2, dim=1)
            first_predicts1 = predicts1[:, 0]
            first_predicts2 = predicts2[:, 0]
            same_pred = torch.eq(first_predicts1, first_predicts2)
            notsame_pred = ~same_pred
            softmax_out1 =  F.softmax(output1, dim=-1)
            softmax_out2 = F.softmax(output2, dim=-1)
            confidence1 = softmax_out1.max(dim=-1)[0]
            confidence2 = softmax_out2.max(dim=-1)[0]
            wj_score_sum += torch.mul(confidence1[same_pred], confidence2[same_pred]).sqrt().sum()/(len(output1))

            ## I_overlap sum
            I_overlap_score_sum += torch.mul(softmax_out1[same_pred], softmax_out2[same_pred]).sum(dim=-1).sqrt().sum()/(len(output1))

            ## PN_I_overlap_
            p_overlap = torch.mul(softmax_out1[same_pred], softmax_out2[same_pred]).sum(dim=-1).sqrt()
            n_overlap = torch.mul(softmax_out1[notsame_pred], softmax_out2[notsame_pred]).sum(dim=-1).sqrt()
            pnI_overlap_score_sum += p_overlap.mean()-(0.5-n_overlap).mean()

            ## PN_softAUoverlap
            aus_matrix = torch.mm(nn.functional.normalize(valid_aus), nn.functional.normalize(valid_aus).transpose(1,0))
            value_min, indices_min = aus_matrix.min(dim=1)
            aus_matrix = aus_matrix - aus_matrix.diag().diag()
            value_max, indices_max = aus_matrix.max(dim=1)
            
            # print(value_max[:50])
            # print(indices_max[:50])
            # print(raw_labels[aus_valid][:50], raw_labels[aus_valid][indices_max][:50])
            # assert(0)

            ausP_EI_sum += torch.mul(confidence1[aus_valid], confidence1[aus_valid][indices_max]).sum(dim=-1).sqrt().mean()
            ausN_EI_sum += torch.mul(confidence1[aus_valid], confidence1[aus_valid][indices_min]).sum(dim=-1).sqrt().mean()

            sp_aus, sn_aus = convert_label_to_AUsim(nn.functional.normalize(real_aus[aus_valid]), raw_labels[aus_valid])
            p_auDis, n_auDis = convert_label_to_intensDis(nn.functional.normalize(real_aus[aus_valid]), raw_labels[aus_valid])
            p_overlap, n_overlap = convert_label_to_overlap(softmax_out1[aus_valid], raw_labels[aus_valid])

            # compute ver acc
            for i in range(len(ver_correct_sum)):
                p_index = (sp_aus < ver_p_ausT[i]).logical_and(sp_aus > ver_p_ausT[i+1])
                p_index[p_auDis.sort()[1][:-int(0.5*len(p_auDis))]] = False
                n_index = (sn_aus > ver_n_ausT[i]).logical_and(sn_aus > ver_n_ausT[i+1]).logical_and(n_auDis>0.5)  
                n_index[n_auDis.sort()[1][:-int(0.5*len(n_auDis))]] = False
                ver_correct_sum[i] += (p_overlap[p_index] > 0.5).sum() + (n_overlap[n_index] < 0.5).sum()
                ver_num[i] += p_index.sum() + n_index.sum()

            # compute tpr at pfr
            hard_p_index = (sp_aus < 0.4) 
            hard_p_index[p_auDis.sort()[1][:-int(0.5*len(p_auDis))]] = False
            hard_n_index = (sn_aus > 0.6)
            hard_n_index[n_auDis.sort()[1][:-int(0.5*len(n_auDis))]] = False
            ver_score.append(torch.cat((p_overlap[hard_p_index], n_overlap[hard_n_index])))
            ver_label.append(torch.cat((torch.ones(hard_p_index.sum()), torch.zeros(hard_n_index.sum()))))    

            loss = nn.CrossEntropyLoss()(output1, raw_labels)

            iter_cnt += 1
            _, predicts = torch.max(output1, 1)
            predicted.append(predicts.cpu().detach())
            gt.append(raw_labels.cpu().detach())
            correct_num = torch.eq(predicts, raw_labels).sum()
            correct_sum += correct_num

            running_loss += loss
            data_num += len(raw_labels)

        running_loss = running_loss / iter_cnt
        test_acc = correct_sum.float() / float(data_num)
        for i in range(len(ver_correct_sum)):
            ver_acc_hard[i] = (ver_correct_sum[i].float() / float(ver_num[i])).item()

        ver_score = torch.cat(ver_score).cpu().detach().numpy()
        ver_label = torch.cat(ver_label).cpu().detach().numpy()

        vr_roc = test_tprATfpr(ver_score, ver_label)
        ver_AUC = roc_auc(ver_score, ver_label)

    return test_acc.item(), ver_acc_hard, vr_roc, ver_AUC, wj_score_sum.item()/iter_cnt, I_overlap_score_sum.item()/iter_cnt, pnI_overlap_score_sum.item()/iter_cnt, \
        ausP_EI_sum.item()/iter_cnt, (ausP_EI_sum - ausN_EI_sum).item()/iter_cnt




def main(args):
    setup_seed(0)

    # torch.cuda.set_device(args.local_rank)
    train_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        # transforms.Grayscale(num_output_channels=3),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(scale=(0.02, 0.25)),
    ])

    eval_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])])
    wjscore_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Grayscale(num_output_channels=3),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    ganimation_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize([128, 128]),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    DaGAN_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize([256, 256]),
        transforms.ToTensor()
    ])


    train_dataset = RafDataset(args, phase='train', transform=train_transforms)
    # train_dataset = RafDataset_twoTransforms(args, phase='train', transform=train_transforms,\
    #                                          transform2=ganimation_transforms)
    test_dataset = AffectData_twoTransforms(args, phase='test', transform=eval_transforms,\
                                            transform2=ganimation_transforms, transform3=DaGAN_transforms)
    wj_test_dataset = AffectData_twoTransforms(args, phase='test', transform=eval_transforms,\
                                            transform2=wjscore_transforms, transform3=DaGAN_transforms)                                         
    # erase_dataset = RafDataset(args, phase='train', transform=ganimation_transforms)

    # train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)


    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.batch_size_fer,
                                               shuffle=True,
                                               num_workers=args.workers,
                                               pin_memory=True,
                                               drop_last=True)
    
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size_fer,
                                              shuffle=True,
                                              num_workers=args.workers,
                                              pin_memory=True)
    wj_test_loader = torch.utils.data.DataLoader(wj_test_dataset, batch_size=args.batch_size_fer,
                                              shuffle=True,
                                              num_workers=args.workers,
                                              pin_memory=True)
    # erase_loader = torch.utils.data.DataLoader(erase_dataset, batch_size=args.batch_size,
    #                                           shuffle=True,
    #                                           num_workers=args.n_threads,
    #                                           pin_memory=True)


    model_name = args.model_name
    job_id = args.job_id
    print(model_name)
    model = timm.create_model(model_name, pretrained=False, num_classes=7).cuda()
    model = torch.nn.DataParallel(model, device_ids=args.gpu_ids)
    model_job_dir = "/root/paddlejob/workspace/env_run/afs/liuyuchi/ckpt/pretrain_MS1MMaEX"
    model_path = os.path.join(model_job_dir, 'MS1MMaEX_'+model_name, 'epoch_1_checkpoint.pth')
    if not os.path.exists(model_path):
        model_path = os.path.join(model_job_dir, 'MS1MMaEX_'+model_name, 'epoch_1_checkpoint.pth')
    checkpoint = torch.load(model_path)
    model.module.load_state_dict(checkpoint, strict=True)
    ## the model remvoed the classifer
    model_feature =  deepcopy(model)
    model_feature.module.reset_classifier(0)
    device = torch.device('cuda:0')

    # model2 = Model(args).cuda()
    model2 = timm.create_model('resnet18', pretrained=False, num_classes=7).cuda()
    model2 = torch.nn.DataParallel(model2, device_ids=args.gpu_ids)
    checkpoint = torch.load("/root/paddlejob/workspace/env_run/afs/liuyuchi/ckpt/pretrain_MS1MMaEX/MS1MMaEX_resnet18/epoch_1_checkpoint.pth")
    model2.module.load_state_dict(checkpoint, strict=True)

    # models for face change
    ganimation_model = create_model(args)

    # depth_encoder = depth.ResnetEncoder(50, False).cuda()
    # depth_decoder = depth.DepthDecoder(num_ch_enc=depth_encoder.num_ch_enc, scales=range(4)).cuda()
    # loaded_dict_enc = torch.load('DaGAN/depth/models/depth_face_model_Voxceleb2_10w/encoder.pth')
    # loaded_dict_dec = torch.load('DaGAN/depth/models/depth_face_model_Voxceleb2_10w/depth.pth')
    # filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in depth_encoder.state_dict()}
    # depth_encoder.load_state_dict(filtered_dict_enc)
    # depth_decoder.load_state_dict(loaded_dict_dec)
    # depth_encoder.eval()
    # depth_decoder.eval()
    # DaGan_gen, DaGan_detector = load_checkpoints(config_path=args.config_DaGAN, checkpoint_path=args.checkpoint_DaGAN)
    # DaGan_gen.eval()
    # DaGan_detector.eval()

    optimizer\
         = torch.optim.Adam(params=[{"params": model.parameters()}], lr=0.0001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)


    # test_acc, test_loss = test(model_ganimation, test_loader, device)
    # print(test_acc, test_loss)
    # assert (0)

    # log_dir = "log/R18MS1M_raf_AU_mixupHardAUs"
    
    log_dir = os.path.join("/root/paddlejob/workspace/env_run/output/", "MS1MMaEX_Affect_VRM")
    args.use_raw_imgs = False
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)

    # pool_manager = my_pool_manager(args, ganimation_model, erase_loader)
    # pool_manager.start()

    # if os.path.exists(os.path.join(log_dir, 'log.txt')):
    #     os.remove(os.path.join(log_dir, 'log.txt')) 
    
    start_time = time.time()
    test_acc, test_acc_small, aus_overlap_mean, aus_PNoverlap_mean, aus_vrm_mean, ensemble_overlap_mean, ensemble_PNoverlap_mean, ensemble_vrm_mean, mixup_overlap_mean, mixup_PNoverlap_mean, mixup_vrm_mean, self_vrm_mean = test(model,model2, model_feature, model, model, model, model, test_loader, eval_transforms, device)
    _, ver_acc_hard, vr_roc, ver_auc, wj_score, I_overlap_score, pnI_overlap_socre, ausP_EI, ausPN_EI = test_wj(model, ganimation_model, wj_test_loader, device)
    print("--- %s seconds ---" % (time.time() - start_time))

    # print(i, train_acc, train_loss)
    dic = {}
    dic['test_acc'] = test_acc
    # dic['test_loss'] = test_loss
    # dic['confidence'] = confidence_mean
    dic['aus_overlap'] = aus_overlap_mean
    dic['aus_PNoverlap'] = aus_PNoverlap_mean
    dic['aus_vrm'] = aus_vrm_mean
    dic['ensemble_overlap'] = ensemble_overlap_mean
    dic['ensemble_PNoverlap'] = ensemble_PNoverlap_mean
    dic['ensemble_vrm'] = ensemble_vrm_mean
    dic['mixup_overlap'] = mixup_overlap_mean
    dic['mixup_PNoverlap'] = mixup_PNoverlap_mean
    dic['mixup_vrm'] = mixup_vrm_mean
    dic['self_vrm_mean'] = self_vrm_mean

    dic['wj_score'] = wj_score
    dic['I_overlap_score'] = I_overlap_score
    dic['pnI_overlap_score'] = pnI_overlap_socre
    dic['ausP_EI'] = ausP_EI
    dic['ausPN_EI'] = ausPN_EI
    # dic['wjPlus_score'] = wjPlus_score_mean
    dic['test_acc_small'] = test_acc_small
    dic['ver_acc_hard'] = ver_acc_hard
    dic['vr_roc'] = vr_roc
    dic['ver_auc'] = ver_auc


    print(test_acc, wj_score, aus_overlap_mean, aus_PNoverlap_mean, aus_vrm_mean, ensemble_overlap_mean, ensemble_PNoverlap_mean, ensemble_vrm_mean, mixup_overlap_mean, mixup_PNoverlap_mean, mixup_vrm_mean, self_vrm_mean)
    print(dic['test_acc_small'])
    print(dic['ver_acc_hard'])
    print(dic['vr_roc'])
    print(dic['ver_auc'])
    np.save(os.path.join(log_dir ,model_name+'.npy'), dic)
    

opt = Options().parse()
main(opt)