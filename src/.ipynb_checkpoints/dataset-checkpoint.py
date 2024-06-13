# -*- coding: utf-8 -*-
import os
import cv2
import torch
import torch.utils.data as data
import pandas as pd
import random
from torchvision import transforms, datasets
from utils import *
import pandas as pd
from PIL import Image
import glob
import numpy as np

## pakages for MS1M
from mxnet import recordio
import mxnet as mx
from ImageNetAll import ObjectNet_to_113

Raf2Affect = {'0': 3, '1': 4, '2': 5, '3': 1, '4': 2, '5': 6, '6': 0,}
Affect2Raf = {'0': 6, '1': 3, '2': 4, '3': 0, '4': 1, '5': 2, '6': 5,}

## generation packages
from model_ganimation import create_model
import torch.nn.functional as F
from skimage.transform import resize
from skimage import io, img_as_float32
import imageio
from skimage.transform import resize

# def AUs_Erasing(ganimation_model, imgs):
#     pred_AUs = ganimation_model(imgs)
#     erased_Aus = pred_AUs.

def AUs_dropout(ganimation_model, imgs):
    pred_AUs = ganimation_model(imgs)
    erased_Aus = F.dropout(pred_AUs, p=0.5)
    erased_imgs = ganimation_model(erased_Aus, imgs)
    return erased_imgs

def get_ImageNetV2(dataset_folder, transform):
    dataset = datasets.ImageFolder(root=dataset_folder, transform=transform)
    return dataset

class ImageNetPredictions_gt(data.Dataset):
    def __init__(self, ImageNetV2_path, prediction_path1, prediction_path2, type='grey'):
        self.ImageNetV2_path = ImageNetV2_path
        self.prediction_path1 = prediction_path1
        self.prediction_path2 = prediction_path2
        self.type = type
        self.prediction_dic1 = np.load(self.prediction_path1, allow_pickle=True).item()
        self.prediction_dic2 = np.load(self.prediction_path2, allow_pickle=True).item()
        self.acc = self.prediction_dic1['acc'].item()
        self.keys = self.prediction_dic1.keys()
        
        self.label = []
        dirs = os.listdir(self.ImageNetV2_path)
        dirs.sort()
        label_flag = 0
        for dir in dirs:
            dir_path = os.path.join(self.ImageNetV2_path, dir)
            if not os.path.isdir(dir_path):
                continue
            img_names = os.listdir(dir_path)
            for img in img_names:
                if 'imagenetv2' in self.ImageNetV2_path:
                    self.label.append(int(dir))
                else:
                    self.label.append(label_flag)
            label_flag+=1

    def __len__(self):
        return len(self.keys)-1

    def __getitem__(self, idx):
        list1 = self.prediction_dic1[str(torch.tensor(idx))]
        list2 = self.prediction_dic2[str(torch.tensor(idx))]
        gt = self.label[idx]
        if self.type == 'grey':
            prediction1_1 = list1[0]
            prediction1_2 = list1[1]
            prediction2_1 = list2[0]
            prediction2_2 = list2[1]
            return prediction1_1, prediction1_2, prediction2_1, prediction2_2, self.acc, gt, idx
        elif self.type == 'mixup':
            prediction1_1 = list1[0]
            prediction1_2 = list1[1]
            lamd = list1[2]
            mix_idx = list1[3].item()
            prediction2_1 = list2[0]
            prediction2_2 = list2[1]
            return prediction1_1, prediction1_2, prediction2_1, prediction2_2, lamd, mix_idx, self.acc, gt, idx



class ImageNetPredictions(data.Dataset):
    def __init__(self, ImageNetV2_path, prediction_path1, prediction_path2, type='grey', set_portion=1):
        self.ImageNetV2_path = ImageNetV2_path
        self.prediction_path1 = prediction_path1
        self.prediction_path2 = prediction_path2
        self.type = type
        self.prediction_dic1 = np.load(self.prediction_path1, allow_pickle=True).item()
        self.prediction_dic2 = np.load(self.prediction_path2, allow_pickle=True).item()
        self.acc = self.prediction_dic1['acc'].item()
        self.keys = self.prediction_dic1.keys()
        self.set_portion = set_portion
        self.label = []
        # dirs = os.listdir(self.ImageNetV2_path)
        # dirs.sort()
        # label_flag = 0
        # for dir in dirs:
        #     dir_path = os.path.join(self.ImageNetV2_path, dir)
        #     if not os.path.isdir(dir_path):
        #         continue
        #     img_names = os.listdir(dir_path)
        #     for img in img_names:
        #         if 'imagenetv2' in self.ImageNetV2_path:
        #             self.label.append(int(dir))
        #         else:
        #             self.label.append(label_flag)
        #     label_flag+=1

    def __len__(self):
        return int((len(self.keys)-1)*(self.set_portion))

    def __getitem__(self, idx):
        # # cifar10 test 补丁
        # list1 = self.prediction_dic1[str(idx)]
        # list2 = self.prediction_dic2[str(idx)]

        list1 = self.prediction_dic1[str(torch.tensor(idx))]
        list2 = self.prediction_dic2[str(torch.tensor(idx))]
        # gt = self.label[idx]
        if self.type == 'grey':
            prediction1_1 = list1[0]
            prediction1_2 = list1[1]
            prediction2_1 = list2[0]
            prediction2_2 = list2[1]
            return prediction1_1, prediction1_2, prediction2_1, prediction2_2, self.acc, idx
        elif self.type == 'mixup':
            prediction1_1 = list1[0]
            prediction1_2 = list1[1]
            lamd = list1[2]
            mix_idx = list1[3].item()
            prediction2_1 = list2[0]
            prediction2_2 = list2[1]
            return prediction1_1, prediction1_2, prediction2_1, prediction2_2, lamd, mix_idx, self.acc, idx

class ImageNetV(data.Dataset):
    def __init__(self, ImageNetV2_path, transform=None, without_aug=False):
        self.ImageNetV2_path = ImageNetV2_path
        self.transform1 = transform
        self.file_paths = []
        self.label = []
        dirs = os.listdir(self.ImageNetV2_path)
        dirs.sort()
        label_flag = 0
        for dir in dirs:
            dir_path = os.path.join(self.ImageNetV2_path, dir)
            if not os.path.isdir(dir_path):
                continue
            img_names = os.listdir(dir_path)
            for img in img_names:       
                if 'imagenetv2' in self.ImageNetV2_path:
                    self.label.append(int(dir))
                elif 'objectnet' in self.ImageNetV2_path:
                    if label_flag in ObjectNet_to_113.keys():
                        self.label.append(ObjectNet_to_113[label_flag])
                    else: 
                        break
                else:
                    self.label.append(label_flag)
                img_path = os.path.join(dir_path, img)
                self.file_paths.append(img_path)
            label_flag+=1

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):

        label = self.label[idx]
        image = cv2.imread(self.file_paths[idx])

        image = image[:, :, ::-1]

        image1 = self.transform1(image)

        return image1, label


class ImageNetV_twoTransforms(data.Dataset):
    def __init__(self, ImageNetV2_path, transform=None, transform2=None, without_aug=False):
        self.ImageNetV2_path = ImageNetV2_path
        self.transform1 = transform
        self.transform2 = transform2
        self.file_paths = []
        self.label = []
        dirs = os.listdir(self.ImageNetV2_path)
        dirs.sort()
        label_flag = 0
        for dir in dirs:
            dir_path = os.path.join(self.ImageNetV2_path, dir)
            if not os.path.isdir(dir_path):
                continue
            img_names = os.listdir(dir_path)
            for img in img_names:       
                if 'imagenetv2' in self.ImageNetV2_path:
                    self.label.append(int(dir))
                elif 'objectnet' in self.ImageNetV2_path:
                    if label_flag in ObjectNet_to_113.keys():
                        self.label.append(ObjectNet_to_113[label_flag])
                    else: 
                        break
                else:
                    self.label.append(label_flag)
                img_path = os.path.join(dir_path, img)
                self.file_paths.append(img_path)
            label_flag+=1

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        try:
            label = self.label[idx]
            image = cv2.imread(self.file_paths[idx])

            image = image[:, :, ::-1]

            image2 = self.transform2(image)

            image1 = self.transform1(image)
        except:
            print(self.file_paths[idx])
            assert(0)


        return image1, image2, label, idx

class iwildcam_twoTransforms(data.Dataset):
    def __init__(self, ImageNetV2_path, transform=None, transform2=None, without_aug=False):
        self.ImageNetV2_path = ImageNetV2_path
        self.transform1 = transform
        self.transform2 = transform2
        self.file_paths = []
        self.label = []
        dirs = os.listdir(self.ImageNetV2_path)
        dirs.sort()
        label_flag = 0
        self.metadata_path = '/root/paddlejob/workspace/env_run/data/iwildcam_v2.0/metadata.csv'
        self.metadata_pd = pd.read_csv(self.metadata_path) 
        self.test_pd = self.metadata_pd[self.metadata_pd['split']=='id_val']
        self._n_classes = max(self.metadata_pd['y']) + 1
        assert len(np.unique(self.metadata_pd['y'])) == self._n_classes

        for index, row in self.test_pd.iterrows():
            filename = row['filename']
            y = row['y']
            self.label.append(y)
            self.file_paths.append(os.path.join(self.ImageNetV2_path, filename))

    def __len__(self):
        # print(len(self.file_paths))
        # assert(0)
        return len(self.file_paths)

    def __getitem__(self, idx):

        label = self.label[idx]
        image = cv2.imread(self.file_paths[idx])

        image = image[:, :, ::-1]

        image2 = self.transform2(image)

        image1 = self.transform1(image)

        return image1, image2, label, idx


class cifar10_twoTransforms(data.Dataset):
    def __init__(self, ImageNetV2_path, transform=None, transform2=None, without_aug=False):
        self.ImageNetV2_path = ImageNetV2_path
        self.transform1 = transform
        self.transform2 = transform2
        self.file_paths = []
        self.label = []
        dirs = os.listdir(self.ImageNetV2_path)
        dirs.sort()
        label_flag = 0
        for dir in dirs:
            dir_path = os.path.join(self.ImageNetV2_path, dir)
            if not os.path.isdir(dir_path):
                continue
            img_names = os.listdir(dir_path)
            for img in img_names:       
                self.label.append(label_flag)
                img_path = os.path.join(dir_path, img)
                self.file_paths.append(img_path)
            label_flag+=1

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):

        label = self.label[idx]
        image = cv2.imread(self.file_paths[idx])

        image = image[:, :, ::-1]

        image2 = self.transform2(image)

        image1 = self.transform1(image)

        return image1, image2, label, idx

class cifar10_C_twoTransforms(data.Dataset):
    def __init__(self, ImageNetV2_path, transform=None, transform2=None, without_aug=False):
        self.ImageNetV2_path = ImageNetV2_path
        self.transform1 = transform
        self.transform2 = transform2
        self.file_paths = []
        self.data = np.load(os.path.join(self.ImageNetV2_path,'gaussian_blur.npy'))[20000:30000]
        self.label = np.load(os.path.join(self.ImageNetV2_path, 'labels.npy'))[20000:30000]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        label = self.label[idx]
        image = self.data[idx]


        image2 = self.transform2(image)

        image1 = self.transform1(image)

        return image1, image2, label, idx

class cifar10_v1_twoTransforms(data.Dataset):
    def __init__(self, ImageNetV2_path, transform=None, transform2=None, without_aug=False):
        self.ImageNetV2_path = ImageNetV2_path
        self.transform1 = transform
        self.transform2 = transform2
        self.file_paths = []
        self.data = np.load(os.path.join(self.ImageNetV2_path,'cifar10.1_v4_data.npy'))
        self.label = np.load(os.path.join(self.ImageNetV2_path, 'cifar10.1_v4_labels.npy'))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        label = self.label[idx]
        image = self.data[idx]

        image2 = self.transform2(image)

        image1 = self.transform1(image)

        return image1, image2, label, idx

class RafDataset(data.Dataset):
    def __init__(self, args, phase, basic_aug=True, transform=None, without_aug=False):
        self.raf_path = args.raf_path
        self.phase = phase
        self.basic_aug = basic_aug
        self.transform = transform
        self.without_aug = without_aug
        df = pd.read_csv(os.path.join(self.raf_path, 'EmoLabel', args.label_path), sep=' ', header=None)

        name_c = 0
        label_c = 1
        if phase == 'train':
            dataset = df[df[name_c].str.startswith('train')]
        else:
            df = pd.read_csv(os.path.join(self.raf_path, 'EmoLabel/list_patition_label.txt'), sep=' ', header=None)
            dataset = df[df[name_c].str.startswith('test')]

        self.label = dataset.iloc[:, label_c].values - 1
        images_names = dataset.iloc[:, name_c].values
        self.aug_func = [flip_image, add_g]
        self.file_paths = []
        self.clean = (args.label_path == 'list_patition_label.txt')

        for f in images_names:
            f = f.split(".")[0]
            f += '_aligned.jpg'
            file_name = os.path.join(self.raf_path, 'Image/aligned', f)
            self.file_paths.append(file_name)

        # read AUs from w
        self.aus_dir = os.path.join(self.raf_path, 'raf_train')


    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        label = self.label[idx]
        image = cv2.imread(self.file_paths[idx])

        image = image[:, :, ::-1]


        if not self.clean:
            image1 = image
            image1 = self.aug_func[0](image)
            image1 = self.transform(image1)

        if self.phase == 'train' and not self.without_aug:
            if self.basic_aug and random.uniform(0, 1) > 0.5:
                image = self.aug_func[1](image)

        if self.transform is not None:
            image = self.transform(image)

        if self.clean:
            image1 = transforms.RandomHorizontalFlip(p=1)(image)

        return image, label, idx

class RafDataset_twoTransforms(data.Dataset):
    def __init__(self, args, phase, basic_aug=True, transform=None, transform2=None, transform3=None):
        self.raf_path = args.raf_path
        self.phase = phase
        self.basic_aug = basic_aug
        self.transform1 = transform
        self.transform2 = transform2
        self.transform3 = transform3
        df = pd.read_csv(os.path.join(self.raf_path, 'EmoLabel', args.label_path), sep=' ', header=None)

        name_c = 0
        label_c = 1
        if phase == 'train':
            dataset = df[df[name_c].str.startswith('train')]
        else:
            # dataset = df[df[name_c].str.startswith('train')]
            df = pd.read_csv(os.path.join(self.raf_path, 'EmoLabel/list_patition_label.txt'), sep=' ', header=None)
            dataset = df[df[name_c].str.startswith('test')]

        self.label = dataset.iloc[:, label_c].values - 1
        images_names = dataset.iloc[:, name_c].values
        self.aug_func = [flip_image, add_g]
        self.file_paths = []
        self.clean = (args.label_path == 'list_patition_label.txt')

        for f in images_names:
            f = f.split(".")[0]
            f += '_aligned.jpg'
            file_name = os.path.join(self.raf_path, 'Image/aligned', f)
            self.file_paths.append(file_name)

        # # random.seed(10)
        # # self.file_paths = random.sample(self.file_paths, len(self.file_paths))
        # if phase == 'train':
        #     self.file_paths = self.file_paths[0:-1]
        #
        # else:
        #     self.file_paths = self.file_paths[0:-1]

        # read AUs from w

        if phase == 'train':
            self.aus_dir = os.path.join(self.raf_path, 'raf_train')
        else:
            self.aus_dir = os.path.join(self.raf_path, 'raf_test')

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):

        label = self.label[idx]
        image = cv2.imread(self.file_paths[idx])

        image = image[:, :, ::-1]

        image2 = self.transform2(image)


        # image3 = imageio.imread(self.file_paths[idx])
        # image3 = resize(image3, (256, 256))[..., :3]
        # image3 = torch.tensor(image3.astype(np.float32)).permute(2, 0, 1)

        if self.phase == 'train':
            if self.basic_aug and random.uniform(0, 1) > 0.5:
                image = self.aug_func[1](image)


        image1 = self.transform1(image)

        # read aus
        partition, id, _ = self.file_paths[idx].split('/')[-1].split('_')
        name = partition+'_'+id
        aus_path = os.path.join(self.aus_dir, str(label), name+'.csv')

        if os.path.exists(aus_path):
            frames_aus = pd.read_csv(aus_path, header=None)
            aus = torch.FloatTensor(frames_aus.iloc[1][5:22].values.astype(np.float)) / 5
            aus_valid = True
            if aus.mean() == 0:
                aus_valid = False
        else:
            aus = torch.FloatTensor(torch.zeros(17)) - 1
            aus_valid = False

        # return image1, image2, image3, aus, aus_valid, label, idx,
        return image1, image2, aus, aus_valid, label, idx,

class CKDataset(data.Dataset):
    def __init__(self, args, phase, basic_aug=True, transform=None):
        self.MaEX_folder = args.MaEX_folder_path
        self.phase = phase
        self.basic_aug = basic_aug
        self.transform = transform
        self.aug_func = [flip_image, add_g]
        self.clean = (args.label_path == 'list_patition_label.txt')

        self.file_paths = []

        self.exp_classes = {'Surprise':0, 'Fear':1, 'Disgust':2, 'Happiness':3, \
                'Sadness':4, 'Anger':5, 'Neutral':6}
        g = os.walk(self.MaEX_folder)
        for path, dir_list, file_list in g:
            for dir_name in dir_list:
                class_name = dir_name.split('_')[-1]
                if class_name in self.exp_classes.keys():
                    self.file_paths.append([os.path.join(self.MaEX_folder, dir_name, '1.jpg'),\
                                            self.exp_classes[class_name]])


    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):

        image = cv2.imread(self.file_paths[idx][0])
        label = self.file_paths[idx][1]

        image = image[:, :, ::-1]

        if not self.clean:
            image1 = image
            image1 = self.aug_func[0](image)
            image1 = self.transform(image1)

        if self.phase == 'train':
            if self.basic_aug and random.uniform(0, 1) > 0.5:
                image = self.aug_func[1](image)

        if self.transform is not None:
            image = self.transform(image)

        if self.clean:
            image1 = transforms.RandomHorizontalFlip(p=1)(image)

        return image, label, idx


class MaEXDataset(data.Dataset):
    def __init__(self, args, phase, basic_aug=True, transform=None):
        self.MaEX_folder = args.MaEX_folder_path
        self.phase = phase
        self.basic_aug = basic_aug
        self.transform = transform
        self.aug_func = [flip_image, add_g]
        self.clean = (args.label_path == 'list_patition_label.txt')

        self.file_paths = []

        self.exp_classes = {'Surprise':0, 'Fear':1, 'Disgust':2, 'Happiness':3, \
                'Sadness':4, 'Anger':5, 'Neutral':6}
        g = os.walk(self.MaEX_folder)
        for path, dir_list, file_list in g:
            for dir_name in dir_list:
                class_name = dir_name.split('_')[-1]
                if class_name in self.exp_classes.keys():
                    self.file_paths.append([os.path.join(self.MaEX_folder, dir_name, '1.jpg'),\
                                            self.exp_classes[class_name]])
                else:
                    print(class_name)
                    assert (0)


    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):

        image = cv2.imread(self.file_paths[idx][0])
        label = self.file_paths[idx][1]

        image = image[:, :, ::-1]

        if not self.clean:
            image1 = image
            image1 = self.aug_func[0](image)
            image1 = self.transform(image1)

        if self.phase == 'train':
            if self.basic_aug and random.uniform(0, 1) > 0.5:
                image = self.aug_func[1](image)

        if self.transform is not None:
            image = self.transform(image)

        if self.clean:
            image1 = transforms.RandomHorizontalFlip(p=1)(image)

        return image, label, idx


class AffectData(data.Dataset):
    def __init__(self, args, phase, basic_aug=True, transform=None):
        self.AffectNet_path = args.AffectNet_path
        self.phase = phase
        self.basic_aug = basic_aug
        if phase == 'train':
            path = os.path.join(self.AffectNet_path, 'filter_train.csv')
        else:
            path = os.path.join(self.AffectNet_path, 'filter_valid.csv')
        self.Affect_dir = os.path.join(args.AffectNet_path, 'aligned')
        data_df_all = pd.read_csv(path)
        self.data_df = data_df_all.loc[data_df_all.iloc[:, -3] < 7]
        self.transform = transform
        self.aug_func = [flip_image, add_g]
        self.DATASET_CLASSES = [
            'Neutral',
            'Happiness',
            'Sadness',
            'Surprise',
            'Fear',
            'Disgust',
            'Anger',
            'Contempt'
        ]

        # read AUs from w
        # self.aus_dir = os.path.join(self.raf_path, 'raf_train')

    def __len__(self):
        # if self.phase == 'train':
        #     return 10000
        return self.data_df.shape[0]

    def __getitem__(self,idx):
        path = os.path.join(self.Affect_dir, self.data_df.iloc[idx, 0])
        label = self.data_df.iloc[idx, -3]
        # image = Image.open(path).convert('RGB')
        image = cv2.imread(path)
        image = image[:, :, ::-1]

        if self.phase == 'train':
            if self.basic_aug and random.uniform(0, 1) > 0.5:
                image = self.aug_func[1](image)

        img = self.transform(image)
        label = Affect2Raf[str(label)]
        return img, label, idx

class AffectData_twoTransforms(data.Dataset):
    def __init__(self, args, phase, basic_aug=True, transform=None, transform2=None, transform3=None):
        self.AffectNet_path = args.AffectNet_path
        self.phase = phase
        self.basic_aug = basic_aug
        if phase == 'train':
            path = os.path.join(self.AffectNet_path, 'filter_train.csv')
        else:
            path = os.path.join(self.AffectNet_path, 'filter_valid.csv')
        self.Affect_dir = os.path.join(args.AffectNet_path, 'aligned')
        data_df_all = pd.read_csv(path)
        self.data_df = data_df_all.loc[data_df_all.iloc[:, -3] < 7]
        self.transform = transform
        self.transform2 = transform2
        self.transform3 = transform3
        self.aug_func = [flip_image, add_g]
        self.DATASET_CLASSES = [
            'Neutral',
            'Happiness',
            'Sadness',
            'Surprise',
            'Fear',
            'Disgust',
            'Anger',
            'Contempt'
        ]

        # read AUs from w
        if phase == 'train':
            self.aus_dir = os.path.join(self.AffectNet_path, 'AffectNet_aus_train')
        else:
            self.aus_dir = os.path.join(self.AffectNet_path, 'AffectNet_aus_valid')

    def __len__(self):
        # if self.phase == 'train':
        #     return 10000
        return self.data_df.shape[0]

    def __getitem__(self,idx):
        path = os.path.join(self.Affect_dir, self.data_df.iloc[idx, 0])
        label = self.data_df.iloc[idx, -3]

        # image3 = imageio.imread(path)
        # image3 = resize(image3, (256, 256))[..., :3]
        # image3 = torch.tensor(image3.astype(np.float32)).permute(2, 0, 1)

        # image = Image.open(path).convert('RGB')
        image = cv2.imread(path)
        image = image[:, :, ::-1]

        img2 = self.transform2(image)

        if self.phase == 'train':
            if self.basic_aug and random.uniform(0, 1) > 0.5:
                image = self.aug_func[1](image)

        img1 = self.transform(image)

        # read aus
        name, _ = path.split('/')[-1].split('.')
        aus_path = os.path.join(self.aus_dir, str(label), name + '.csv')

        if os.path.exists(aus_path):
            frames_aus = pd.read_csv(aus_path, header=None)
            aus = torch.FloatTensor(frames_aus.iloc[1][5:22].values.astype(np.float)) / 5
            aus_valid = True
            if aus.mean() == 0:
                aus_valid = False
        else:
            aus = torch.FloatTensor(torch.zeros(17))
            aus_valid = False

        label = Affect2Raf[str(label)]

        # return img1, img2, image3, aus, aus_valid, label, idx,
        return img1, img2, aus, aus_valid, label, idx,


class MS1MMaEX_Dataset(data.Dataset):
    def __init__(self, args, phase, basic_aug=True, transform=None):
        self.dataset_folder = args.MaEX_folder_path
        self.phase = phase
        self.basic_aug = basic_aug
        self.transform = transform
        self.aug_func = [flip_image, add_g]
        self.clean = (args.label_path == 'list_patition_label.txt')

        self.file_paths = []

        self.exp_classes = {'Surprise':0, 'Fear':1, 'Disgust':2, 'Happiness':3, \
                'Sadness':4, 'Anger':5, 'Neutral':6}

        self.file_paths = glob.glob(self.dataset_folder + "/*/*.jpg")


    def __len__(self):
        return len(self.file_paths)
        # return 5000

    def __getitem__(self, idx):

        image = cv2.imread(self.file_paths[idx])
        class_name = self.file_paths[idx].split('_')[-1][:-4]
        label = self.exp_classes[class_name]
        image = image[:, :, ::-1]


        if not self.clean:
            image1 = image
            image1 = self.aug_func[0](image)
            image1 = self.transform(image1)

        if self.phase == 'train':
            if self.basic_aug and random.uniform(0, 1) > 0.5:
                image = self.aug_func[1](image)

        if self.transform is not None:
            image = self.transform(image)

        if self.clean:
            image1 = transforms.RandomHorizontalFlip(p=1)(image)

        return image, label, idx

class MS1MMaEXRec_Dataset(data.Dataset):
    def __init__(self, args, phase, basic_aug=True, transform=None):
        self.dataset_folder = args.MaEX_folder_path
        self.phase = phase
        self.basic_aug = basic_aug
        self.transform = transform
        self.aug_func = [flip_image, add_g]
        self.clean = (args.label_path == 'list_patition_label.txt')

        self.file_paths = []

        self.exp_classes = {'Surprise':0, 'Fear':1, 'Disgust':2, 'Happiness':3, \
                'Sadness':4, 'Anger':5, 'Neutral':6}

        # self.file_paths = glob.glob(self.dataset_folder+"/*/*.jpg")

        path_imgidx = '/home/yuchi/micro-expression/ganimation_replicate/output/MS1MMaEX.idx'
        path_imgrec = '/home/yuchi/micro-expression/ganimation_replicate/output/MS1MMaEX.rec'
        imgrec = recordio.MXIndexedRecordIO(path_imgidx, path_imgrec, 'r')
        header_0, s0 = recordio.unpack(imgrec.read_idx(1))
        print(header_0.label)
        img = mx.image.imdecode(s0).asnumpy()
        print(img.shape)
        img = Image.fromarray(img)
        assert (0)


    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):

        image = cv2.imread(self.file_paths[idx])
        class_name = self.file_paths[idx].split('_')[-1][:-4]
        label = self.exp_classes[class_name]

        image = image[:, :, ::-1]

        if not self.clean:
            image1 = image
            image1 = self.aug_func[0](image)
            image1 = self.transform(image1)

        if self.phase == 'train':
            if self.basic_aug and random.uniform(0, 1) > 0.5:
                image = self.aug_func[1](image)

        if self.transform is not None:
            image = self.transform(image)

        if self.clean:
            image1 = transforms.RandomHorizontalFlip(p=1)(image)

        return image, label, idx