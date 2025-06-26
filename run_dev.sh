#!/bin/bash
#SBATCH -p gpu_devel
#SBATCH --gres=gpu:a100.MIG.20gb:1

#SBATCH --job-name=tk_module_addition_feature # Job name
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=10g
#SBATCH --time=12:00:00
#SBATCH --output=slurm_output/%j.out 
#SBATCH --error=slurm_output/%j.err 
#SBATCH --requeue 

echo '-------------------------------'
cd ${SLURM_SUBMIT_DIR}
echo ${SLURM_SUBMIT_DIR}
echo Running on host $(hostname)
echo Time is $(date)
echo SLURM_NODES are $(echo ${SLURM_NODELIST})
echo '-------------------------------'
echo -e '\n\n'

export PROCS=${SLURM_CPUS_ON_NODE}

# Set the working directory
cd /home/jh3439/Grokking

module load CUDA
module load cuDNN
module load miniconda
conda activate envs_LARA
python exp_grokk.py