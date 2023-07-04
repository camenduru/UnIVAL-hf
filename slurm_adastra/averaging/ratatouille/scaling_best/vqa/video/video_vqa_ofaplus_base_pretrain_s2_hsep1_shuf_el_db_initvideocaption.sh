#!/bin/bash
   
#SBATCH --job-name=video_vqa_ofaplus_base_pretrain_s2_hsep1_shuf_el_db_initvideocaption
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --gpus=16
#SBATCH --threads-per-core=2
#SBATCH --gpu-bind=closest 
####SBATCH --nodelist=x1004c4s1b0n0,x1004c4s1b1n0
#SBATCH --time=24:00:00
#SBATCH -C MI250
#SBATCH -A gda2204
#SBATCH --mail-type=END,FAIL
#SBATCH --output=/lus/home/NAT/gda2204/mshukor/logs/slurm/video_vqa_ofaplus_base_pretrain_s2_hsep1_shuf_el_db_initvideocaption.out
#SBATCH --exclusive
#SBATCH --mail-user=mustafa.shukor@isir.upmc.fr


cd /lus/home/NAT/gda2204/mshukor/code/ofa_ours/run_scripts
source /lus/home/NAT/gda2204/mshukor/.bashrc

conda activate main
 

rm core-python3*


srun -l -N 2 -n 2 -c 128 --gpus=16 --gpu-bind=closest bash averaging/ratatouille/scaling_best/vqa/video/video_vqa_ofaplus_base_pretrain_s2_hsep1_shuf_el_db_initvideocaption.sh


