#!bin/bash

# 'resmlp_36_224' 'cait_s24_224' 'convit_base' 'convit_tiny' 'twins_pcpvt_base' \
# 'eca_nfnet_l1' 'xcit_tiny_24_p8_384_dist' 'efficientnet_b1' 'efficientnet_b3' 'efficientnet_b4' \
# 'tf_efficientnet_b2' 'tf_efficientnet_lite1' 'convnext_base' 'convnext_small' 'resnetrs350' \
# 'pit_xs_distilled_224' 'crossvit_small_240' 'tinynet_e' 'tinynet_d' 'repvgg_b2g4' \

# 'mnasnet_small' 'dla46x_c' 'lcnet_050' 'tv_resnet34' 'tv_resnet50' 'tv_resnet101' \
# 'tv_resnet152' 'densenet121' 'inception_v4' 'resnet26d' 'mobilenetv2_140' 'hrnet_w40' \
# 'xception' 'xception41' 'resnet18' 'resnet34' 'seresnet50' 'mobilenetv2_050' 'seresnet33ts' \
# 'wide_resnet50_2' 'wide_resnet101_2' 'resnet18d' 'hrnet_w18_small' 'gluon_resnet152_v1d' \
# 'hrnet_w48' 'hrnet_w44' 'repvgg_b2' 'densenet201' 'hrnet_w18_small' 'resnet101d' 

# 'gluon_resnet101_v1d' 'gluon_resnet101_v1s' 'gluon_xception65' 'gluon_seresnext50_32x4d' 'gluon_senet154' \
# 'gluon_inception_v3' 'gluon_resnet101_v1c' 'tf_inception_v3' 'tv_densenet121' \
# 'tv_resnext50_32x4d' 'repvgg_b1g4' 'resnext26ts' 'ghostnet_100' 'crossvit_9_240' 'rexnet_150' 'rexnet_130' 'resnetrs50' 'resnet50d' 'resnet50' \

# 'resnetv2_50' 'resnetrs152' 'resnetrs101' 'dpn92' 'dpn98' 'dpn68' 'vgg19_bn' 'vgg16_bn' \
# 'vgg13_bn' 'vgg11_bn' 'vgg11' 'vgg11_bn' 'vgg16' 'vgg19' 'swin_small_patch4_window7_224' 'deit_base_patch16_224' 'deit_small_distilled_patch16_224' \
# 'densenet161' 'tf_mobilenetv3_large_075' 'inception_v3'

#cifar10
# "DenseNet121" "DenseNet169" "DenseNet201" "DenseNet161" "densenet_cifar" "DLA" "SimpleDLA" "DPN26" "DPN92" "EfficientNetB0" \ 
# "GoogLeNet" "LeNet" "MobileNet" "MobileNetV2" "PNASNetA" "PNASNetB" "PreActResNet18" "PreActResNet34" "PreActResNet50" \ 
# "PreActResNet101" 

# "PreActResNet152" "RegNetX_200MF" "RegNetX_400MF" "RegNetY_400MF" "ResNet18" "ResNet34" "ResNet50" "ResNet101" "ResNet152" \ 
# "ResNeXt29_2x64d" "ResNeXt29_4x64d" "ResNeXt29_8x64d" "ResNeXt29_32x4d" "SENet18" "ShuffleNetG2" "ShuffleNetG3" "ShuffleNetV2" \ 
# "VGG11" "VGG13" "VGG16" "VGG19"

all_model_names=("ShuffleNetG2" "ShuffleNetG3")


for m_name in ${all_model_names[@]}
  do
  echo $m_name
  CUDA_VISIBLE_DEVICES=0,1,2,3 python train_multiModels_cifar.py --model_name $m_name --label_path 'list_patition_label.txt' --mode test --data_root datasets/celebA \
  --batch_size_fer 256 --gpu_ids 0,1,2,3 --ckpt_dir ../models/ganimation/190327_160828/ --epochs 20
  done
#python -m torch.distributed.launch --nproc_per_node=3 --nnodes=1 --node_rank=0 --master_addr="127.0.0.1" --master_port=12581 main.py
# python main.py --label_path 'noise02.txt'
# python main.py --label_path 'noise03_plus1.txt'

