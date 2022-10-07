#!/bin/bash

#$ -l rt_F=1
#$ -l h_rt=72:00:00
#$ -j y
#$ -cwd
#$ -N tran_pcdre
#$ -o /groups/gaa50131/user/qiuyue/3d_change_captioning/3drepresentation/re01

#source /etc/profile.d/modules.sh
#module load cuda/11.1/11.1.1
#module load cudnn/8.2/8.2.1
#module load gcc/9/3/0
#module load cuda/9.2/9.2.88.1
#module load cudnn/7.4/7.4.2
#module load nccl/2.3/2.3.5-2
#module load openmpi/2.1.5

#ANACONDA_HOME := $(HOME)/anaconda

export PATH="/home/aab11336im/anaconda3/bin:${PATH}"

#cp -r /groups/gaa50131/datasets/sunrgbd1/sunrgbd_pc_bbox_votes_50k_v1_train $SGE_LOCALDIR
#cp -r /groups/gaa50131/datasets/sunrgbd1/sunrgbd_pc_bbox_votes_50k_v1_val $SGE_LOCALDIR
source activate sttran_new_1

cd /groups/gaa50131/user/qiuyue/3d_change_captioning/3drepresentation/re01

python main.py --max_epoch 1080 --nqueries 32 --dataset_name ecc1 --checkpoint_dir 'checkpoints/' --use_color
