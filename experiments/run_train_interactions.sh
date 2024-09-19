#!/bin/bash
#SBATCH --gres=gpu:RTX5000:1
#SBATCH -p gpu
#SBATCH -t 2-00:00:00
#SBATCH -o /usr/users/vogg/monkey-tracking-in-the-wild/slurm_files/job-%J.out


cd /usr/users/vogg/monkey-tracking-in-the-wild/src
module load miniforge3
source activate /usr/users/vogg/.conda/envs/mktrack

python train_interactions.py interactions --exp_id lemurs_full\
                    --interaction_data_root '../../fcsgg/'\
                    --interaction_data_file 'lemurs/data/data/pkls/new_lemur_train.pkl'\
                    --interaction_output_folder '../exp/interactions/sum_vectors'\
                    --num_epochs 20\
                    --gpus 0\
                    --store_opt\
                    --lr 0.0001\
                    --batch_size 32\
                    --seed 2\
                    --val_intervals 1\
                    --save_all
cd ..
