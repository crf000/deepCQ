#!/bin/bash
#SBATCH --job-name=seg_qua
#SBATCH -N 1 # 使用1个节点
#SBATCH --gres=gpu:1 # 占用一个gpu
#SBATCH -o %j.log
#SBATCH -e %j.err
#SBATCH -p GPU1 # 占用GPU1
#SBATCH -w gpu05  # 指定节点列表是gpu05
echo $(hostname) $CUDA_VISIBLE_DEVICES

srun /Share/apps/singularity/bin/singularity exec /Share/imgs/ahu_ai.img python /Share/home/E31714025/paper_model/run_nolstm.py

