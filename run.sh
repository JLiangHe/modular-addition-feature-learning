#!/bin/bash

#SBATCH --job-name=tk_module_addition_feature # Job name
#SBATCH --partition=gpu 
#SBATCH --gres=gpu:h100:1 
#SBATCH --qos=qos_zhuoran_yang
#SBATCH --ntasks=1 
#SBATCH --cpus-per-task=16
#SBATCH --time=48:00:00 
#SBATCH --output=slurm_output/%j.out 
#SBATCH --error=slurm_output/%j.err 
#SBATCH --requeue 

echo '-------------------------------'
cd ${SLURM_SUBMIT_DIR}
echo ${SLURM_SUBMIT_DIR}
echo Running on host $(hostname)
echo Time is $(date)
echo '-------------------------------'
echo -e '\n\n'

export PROCS=${SLURM_CPUS_ON_NODE}

# Set the working directory
cd /gpfs/radev/home/jh3439/module-addition-feature

module load CUDA
module load cuDNN
module load miniconda
conda activate envs_LARA

echo "Starting experiments..."
echo "============================================================="

cd src

#python module_nn.py --init_type random --act_type ReLU --optimizer AdamW --init_scale 0.1
python module_nn.py --init_type random --act_type ReLU --optimizer SGD --lr 0.1 --init_scale 0.01
#python module_nn.py --init_type single-freq --act_type Quad --optimizer SGD --lr 0.1 --init_scale 0.02
#python module_nn.py --init_type single-freq --act_type ReLU --optimizer SGD --lr 0.01 --init_scale 0.002
#python module_nn.py --init_type random --act_type Quad --optimizer SGD --lr 0.1 --init_scale 0.1
