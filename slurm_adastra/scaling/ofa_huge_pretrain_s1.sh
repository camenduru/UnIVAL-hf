#!/bin/bash
   
#SBATCH --job-name=ofa_huge_pretrain_s1
#SBATCH --nodes=32
#SBATCH --ntasks=32
###SBATCH --gpus=128
#SBATCH --gpus-per-node=4
#SBATCH --threads-per-core=2
#SBATCH --gpu-bind=closest 
####SBATCH --nodelist=x1004c7s2b1n0,x1004c7s3b0n0,x1004c7s3b1n0,x1004c7s4b0n0
#SBATCH --mail-type=END,FAIL
#SBATCH --output=/lus/home/NAT/gda2204/mshukor/logs/slurm/ofa_huge_pretrain_s1.out
#SBATCH --exclusive
#SBATCH --time=24:00:00
#SBATCH -C MI250
#SBATCH -A gda2204
#SBATCH --cpus-per-task=128



#SBATCH --mail-user=mustafa.shukor@isir.upmc.fr


cd /lus/home/NAT/gda2204/mshukor/code/ofa_ours/run_scripts
source /lus/home/NAT/gda2204/mshukor/.bashrc

conda activate main
 

rm core-*


srun -l -N 32 -n 32 -c 128 --gpus=128 --gpu-bind=closest bash pretraining/scaling/ofa_huge_pretrain_s1.sh