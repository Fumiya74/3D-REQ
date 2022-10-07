#!/bin/bash

#$ -l rt_F=1
#$ -l h_rt=72:00:00
#$ -j y
#$ -cwd
#$ -N setup
#$ -o /groups/gab50221/user/qiuyue/change_captioning/3drepresentation/3detr_testgpu/utils

source /etc/profile.d/modules.sh
module load cuda/10.2/10.2.89
module load cudnn/8.2/8.2.1
#module load gcc/9/3/0
#module load cuda/9.2/9.2.88.1
#module load cudnn/7.4/7.4.2
#module load nccl/2.3/2.3.5-2
#module load openmpi/2.1.5

#ANACONDA_HOME := $(HOME)/anaconda

export PATH="/home/aab11336im/anaconda3/bin:${PATH}"

source activate sttran_new_1

cd /groups/gab50221/user/qiuyue/change_captioning/3drepresentation/3detr_testgpu/utils

python cython_compile.py build_ext --inplace

