#!/bin/bash
#SBATCH --gres=gpu:rtx5000:4
#SBATCH -p gpu
#SBATCH -t 2-00:00:00
#SBATCH -o /usr/users/vogg/monkey-tracking-in-the-wild/slurm_files/job-%J.out

cd /local/eckerlab/



if [ ! -d "LemurLabellingNov22" ] 
then
    tar xf /usr/users/agecker/datasets/LemurLabellingNov22.tar
fi

source activate mktrack

cd /usr/users/vogg/monkey-tracking-in-the-wild/src

python train.py mot --exp_id lemur_1500_2\
                    --load_model '../exp/mot/macaquecp_seed1/model_250.pth'\
                    --num_epochs 500\
                    --lr_step 200\
                    --lr '1e-4'\
                    --data_cfg '../src/lib/cfg/lemur_1500.json'\
                    --store_opt\
                    --arch hrnet_32\
                    --gpus 0,1,2,3\
                    --batch_size 8\
                    --data_dir '/local/eckerlab/'\
                    --seed 3\
                    --reid_cls_names 'lemur,box'\
                    --val_intervals 20\
                    --save_all
                    
                    # --resume\
                    #--use_gc\
                    #--gc_cls_names 'Richard,Kiwi,Timo,Alex,Flo'\
                    # --cat_spec_wh\
                    
                    
                    #--trainval\
                    
                    #--train_only_gc '../models/model_120.pth'
cd ..
