#!bin/bash

# 'ssl_resnext101_32x8d' 'ssl_resnext101_32x16d' 'swsl_resnext101_32x8d' \
# 'swsl_resnext101_32x16d' 'ssl_resnext101_32x4d' 'ssl_resnext50_32x4d' 'ssl_resnet50' \
# 'swsl_resnext101_32x4d' 'swsl_resnext50_32x4d' 'swsl_resnet50' 'tf_efficientnet_l2_ns_475' \
# 'tf_efficientnet_b7_ns' 'tf_efficientnet_b6_ns' 'tf_efficientnet_b4_ns' 'tf_efficientnet_b5_ns' \
# 'convnext_xlarge_384_in22ft1k' 'convnext_xlarge_in22ft1k' 'convnext_large_384_in22ft1k' \
# 'convnext_large_in22ft1k' 'convnext_base_384_in22ft1k' 'convnext_base_in22ft1k' 'resnetv2_152x2_bitm' \
# 'resnetv2_152x4_bitm' 'resnetv2_50x1_bitm' 'resmlp_big_24_224_in22ft1k' 'resmlp_big_24_distilled_224' \
# 'tf_efficientnetv2_s_in21ft1k' 'tf_efficientnetv2_m_in21ft1k' 'tf_efficientnetv2_l_in21ft1k' \
# 'tf_efficientnetv2_xl_in21ft1k' 'vit_large_patch16_384' 'swin_large_patch4_window12_384' \
# 'beit_large_patch16_512' 'beit_large_patch16_384' 'beit_large_patch16_224' 'beit_base_patch16_384' \
# 'vit_base_patch16_384' 'vit_small_r26_s32_384' 'vit_tiny_patch16_384' 'vit_large_r50_s32_384' \
# 'mixer_b16_224_miil' 'resmlp_big_24_224' 'resnetv2_50x1_bit_distilled' 'ig_resnext101_32x16d' \
# 'ig_resnext101_32x32d' 'ig_resnext101_32x8d' 'ig_resnext101_32x48d' \
# 'regnety_016' 'regnety_032' \
# 'resmlp_36_224' 'cait_s36_384' 'cait_s24_224' 'convit_base' 'convit_tiny' 'twins_pcpvt_base' 'eca_nfnet_l1' 'xcit_tiny_24_p8_384_dist' \
# 'efficientnet_b1' 'efficientnet_b3' 'efficientnet_b4' 'tf_efficientnet_b2'  'tf_efficientnet_lite1'  'convnext_base'  'convnext_small'  \
# 'resnetrs350' 'pit_xs_distilled_224' 'crossvit_small_240' 'botnet26t_256' 'tinynet_e' 'tinynet_d' 'repvgg_b2g4' 'mnasnet_small' 'dla46x_c' \

# 'lcnet_050' 'tv_resnet34' 'tv_resnet50' 'tv_resnet101' 'tv_resnet152' 'densenet121' 'inception_v4' 'resnet26d' 'mobilenetv2_140' 'hrnet_w40' \
# 'xception' 'xception41' 'resnet18' 'resnet34' 'seresnet50' 'mobilenetv2_050' 'seresnet33ts' 'wide_resnet50_2' 'wide_resnet101_2' 'resnet18d' \
# 'hrnet_w18_small' 'gluon_resnet152_v1d' 'hrnet_w48'  'hrnet_w44'  'repvgg_b2'  'densenet201'  'hrnet_w18_small'  'resnet101d' 'gluon_resnet101_v1d' \
# 'gluon_resnet101_v1s' 'gluon_xception65' 'gluon_seresnext50_32x4d' 'gluon_senet154' 'gluon_inception_v3' 'gluon_resnet101_v1c' 'tf_inception_v3' \
# 'tv_densenet121' 'tv_resnext50_32x4d' 'repvgg_b1g4' 'resnext26ts' 'ghostnet_100' 'crossvit_9_240' 'deit_base_patch16_384' 'rexnet_150' 'rexnet_130' \
# 'resnetrs50' 'resnet50d' 'resnet50' 'resnetv2_50' 'resnetrs152' 'resnetrs101' 'dpn92' 'dpn98' 'dpn68' 'vgg19_bn' 'vgg16_bn' 'vgg13_bn' 'vgg11_bn' \
# 'vgg11' 'vgg11_bn' 'vgg16' 'vgg19' 'swin_small_patch4_window7_224' 'swin_base_patch4_window12_384' 'deit_base_patch16_224' \
# 'deit_small_distilled_patch16_224' 'densenet161' 'tf_mobilenetv3_large_075' 'inception_v3'



# # iwildcamo models:
# 'iwildcam_afn_extraunlabeled_tune0' 'iwildcam_dann_coarse_extraunlabeled_tune0' 'iwildcam_deepcoral_coarse_extraunlabeled_tune0' \ 'iwildcam_deepcoral_coarse_singlepass_extraunlabeled_tune0' 'iwildcam_deepCORAL_seed0' 'iwildcam_deepCORAL_seed1' 'iwildcam_deepCORAL_seed2' \ 'iwildcam_deepCORAL_tune' 'iwildcam_ermaugment_tune0' 'iwildcam_ermoracle_extraunlabeled_tune0' 'iwildcam_erm_seed0' 'iwildcam_erm_seed1' \
# 'iwildcam_erm_seed2' 'iwildcam_erm_tune0' 'iwildcam_erm_tuneA_seed0' 'iwildcam_erm_tuneB_seed0' 'iwildcam_erm_tuneC_seed0' 'iwildcam_erm_tuneD_seed0' \
# 'iwildcam_erm_tuneE_seed0' 'iwildcam_erm_tuneF_seed0' 'iwildcam_erm_tuneG_seed0' 'iwildcam_erm_tuneH_seed0' 'iwildcam_fixmatch_extraunlabeled_tune0' \ 'iwildcam_groupDRO_seed0' 'iwildcam_groupDRO_seed1' 'iwildcam_groupDRO_seed2' 'iwildcam_irm_seed0' 'iwildcam_irm_seed1' 'iwildcam_irm_seed2' 'iwildcam_irm_tune' \ 'iwildcam_noisystudent_extraunlabeled_seed0' 'iwildcam_pseudolabel_extraunlabeled_tune0' 'iwildcam_swav30_ermaugment_seed0' 

## cifar10 models:

# "DenseNet121_20" "DenseNet169_20" "DenseNet201_20" "DenseNet161_20" "densenet_cifar_20" "DLA_20" "SimpleDLA_20" "DPN26_20" "DPN92_20" "EfficientNetB0_20" \ 
# "GoogLeNet_20" "LeNet_20" "MobileNet_20" "MobileNetV2_20" "PNASNetA_20" "PNASNetB_20" "PreActResNet18_20" "PreActResNet34_20" "PreActResNet50_20" \ 
# "PreActResNet101_20" "PreActResNet152_20" "RegNetX_200MF_20" "RegNetX_400MF_20" "RegNetY_400MF_20" "ResNet18_20" "ResNet34_20" "ResNet50_20" "ResNet101_20" "ResNet152_20" \ 
# "ResNeXt29_2x64d_20" "ResNeXt29_4x64d_20" "ResNeXt29_8x64d_20" "ResNeXt29_32x4d_20" "SENet18_20" "ShuffleNetG2_20" "ShuffleNetG3_20" "ShuffleNetV2_20" \ 
# "VGG11_20" "VGG13_20" "VGG16_20" "VGG19_20"


# "DenseNet121" "DenseNet169" "DenseNet201" "DenseNet161" "densenet_cifar" "DLA" "SimpleDLA" "DPN26" "DPN92" "EfficientNetB0" \ 
# "GoogLeNet" "LeNet" "MobileNet" "MobileNetV2" "PNASNetA" "PNASNetB" "PreActResNet18" "PreActResNet34" "PreActResNet50" \ 
# "PreActResNet101" "PreActResNet152" "RegNetX_200MF" "RegNetX_400MF" "RegNetY_400MF" "ResNet18" "ResNet34" "ResNet50" "ResNet101" "ResNet152" \ 
# "ResNeXt29_2x64d" "ResNeXt29_4x64d" "ResNeXt29_8x64d" "ResNeXt29_32x4d" "SENet18" "ShuffleNetG2" "ShuffleNetG3" "ShuffleNetV2" \ 
# "VGG11" "VGG13" "VGG16" "VGG19"

# "cifar10_mobilenetv2_x0_5" "cifar10_mobilenetv2_x0_75" "cifar10_mobilenetv2_x1_0" "cifar10_mobilenetv2_x1_4" "cifar10_repvgg_a0" "cifar10_repvgg_a1" \ "cifar10_repvgg_a2" "cifar10_resnet20" "cifar10_resnet32" "cifar10_resnet44" "cifar10_resnet56" "cifar10_shufflenetv2_x0_5" "cifar10_shufflenetv2_x1_0" \
# "cifar10_shufflenetv2_x1_5" "cifar10_shufflenetv2_x2_0" "cifar10_vgg11_bn" "cifar10_vgg13_bn" "cifar10_vgg16_bn" "cifar10_vgg19_bn"



all_model_names=('ssl_resnext101_32x8d' 'ssl_resnext101_32x16d' 'swsl_resnext101_32x8d' \
'swsl_resnext101_32x16d' 'ssl_resnext101_32x4d' 'ssl_resnext50_32x4d' 'ssl_resnet50' \
'swsl_resnext101_32x4d' 'swsl_resnext50_32x4d' 'swsl_resnet50' 'tf_efficientnet_l2_ns_475' \
'tf_efficientnet_b7_ns' 'tf_efficientnet_b6_ns' 'tf_efficientnet_b4_ns' 'tf_efficientnet_b5_ns' \
'convnext_xlarge_384_in22ft1k' 'convnext_xlarge_in22ft1k' 'convnext_large_384_in22ft1k' \
'convnext_large_in22ft1k' 'convnext_base_384_in22ft1k' 'convnext_base_in22ft1k' 'resnetv2_152x2_bitm' \
'resnetv2_152x4_bitm' 'resnetv2_50x1_bitm' 'resmlp_big_24_224_in22ft1k' 'resmlp_big_24_distilled_224' \
'tf_efficientnetv2_s_in21ft1k' 'tf_efficientnetv2_m_in21ft1k' 'tf_efficientnetv2_l_in21ft1k' \
'tf_efficientnetv2_xl_in21ft1k' 'vit_large_patch16_384' 'swin_large_patch4_window12_384' \
'beit_large_patch16_512' 'beit_large_patch16_384' 'beit_large_patch16_224' 'beit_base_patch16_384' \
'vit_base_patch16_384' 'vit_small_r26_s32_384' 'vit_tiny_patch16_384' 'vit_large_r50_s32_384' \
'mixer_b16_224_miil' 'resmlp_big_24_224' 'resnetv2_50x1_bit_distilled' 'ig_resnext101_32x16d' \
'ig_resnext101_32x32d' 'ig_resnext101_32x8d' 'ig_resnext101_32x48d' \
'regnety_016' 'regnety_032' \
'resmlp_36_224' 'cait_s36_384' 'cait_s24_224' 'convit_base' 'convit_tiny' 'twins_pcpvt_base' 'eca_nfnet_l1' 'xcit_tiny_24_p8_384_dist' \
'efficientnet_b1' 'efficientnet_b3' 'efficientnet_b4' 'tf_efficientnet_b2'  'tf_efficientnet_lite1'  'convnext_base'  'convnext_small'  \
'resnetrs350' 'pit_xs_distilled_224' 'crossvit_small_240' 'botnet26t_256' 'tinynet_e' 'tinynet_d' 'repvgg_b2g4' 'mnasnet_small' 'dla46x_c')


job_id='job-0bb63458ffa013c3'
export PYTHONPATH="/root/paddlejob/workspace/env_run/Erasing-Attention-Consistency/src2/DaGAN/:$PYTHONPATH" 
for m_name in ${all_model_names[@]}
  do
  echo $m_name
  OMP_NUM_THDS=20 CUDA_VISIBLE_DEVICES=0,1,2,3 python test_metrics_V2.py --job_id $job_id --model_name $m_name --label_path 'list_patition_label.txt' --mode test --data_root datasets/celebA --gpu_ids 0,1,2,3 --ckpt_dir ../models/ganimation/190327_160828/ --load_epoch 30 --batch_size_fer 512 --set_portion 1 --out_dir imagenet_a_out_rotation_May24
  done
#python -m torch.distributed.launch --nproc_per_node=3 --nnodes=1 --node_rank=0 --master_addr="127.0.0.1" --master_port=12581 main.py
# python main.py --label_path 'noise02.txt'
# python main.py --label_path 'noise03_plus1.txt'

