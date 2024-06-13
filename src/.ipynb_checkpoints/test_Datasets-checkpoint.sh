#!bin/bash

# 'gaussian_noise' 'shot_noise' 'impulse_noise' 'defocus_blur' 'glass_blur' 'motion_blur' 'zoom_blur' 'frost' 'snow' 'fog' \
# 'brightness' 'elastic_transform' 'pixelate' 'jpeg_compression' 'speckle_noise' 'spatter' 'gaussian_blur' 'saturate'

dataset_names=('gaussian_noise' 'shot_noise' 'impulse_noise' 'defocus_blur' 'glass_blur' 'motion_blur' 'zoom_blur' 'frost' 'snow' 'fog')
levels=('1' '2' '3' '4' '5')
m_name='tf_efficientnet_b2'
dataset_dir='/root/paddlejob/workspace/env_run/data/imagenet-c'

job_id='job-0bb63458ffa013c3'
export PYTHONPATH="/root/paddlejob/workspace/env_run/Erasing-Attention-Consistency/src2/DaGAN/:$PYTHONPATH" 
for d_name in ${dataset_names[@]}
  do
  echo $d_name
  for level in ${levels[@]}
    do 
    echo $level
    dataset_path="$dataset_dir/$d_name/$level"
    echo $dataset_path
    OMP_NUM_THDS=20 CUDA_VISIBLE_DEVICES=2 python test_metrics_sets.py --job_id $job_id --model_name $m_name --label_path 'list_patition_label.txt' --mode test --data_root datasets/celebA --gpu_ids 2 --ckpt_dir ../models/ganimation/190327_160828/ --load_epoch 30 --ImageNetV2_path $dataset_path --batch_size_fer 256
    done
  done
#python -m torch.distributed.launch --nproc_per_node=3 --nnodes=1 --node_rank=0 --master_addr="127.0.0.1" --master_port=12581 main.py
# python main.py --label_path 'noise02.txt'
# python main.py --label_path 'noise03_plus1.txt'
