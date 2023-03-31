#!/bin/bash
#SBATCH --gres=gpu:rtx5000:4
#SBATCH --qos=long
#SBATCH -p gpu
#SBATCH -t 3-04:00:00
#SBATCH -o /usr/users/vogg/monkey-tracking-in-the-wild/slurm_files/job-%J.out

cd /local/eckerlab/

if [ ! -d "MacaquePose" ] 
then
    tar xf /usr/users/agecker/datasets/MacaquePose.tar
fi

source activate mktrack

cd /usr/users/vogg/monkey-tracking-in-the-wild/src
python train.py mot --exp_id macaquepose_dla_seed1\
                    --load_model '../models/hrnetv2_w32_imagenet_pretrained.pth'\
                    --num_epochs 300\
                    --lr_step 200\
                    --lr '1e-4'\
                    --data_cfg '../src/lib/cfg/macaquepose.json'\
                    --store_opt\
                    --arch hrnet_32\
                    --gpus 0,1,2,3\
                    --batch_size 8\
                    --data_dir '/local/eckerlab/'\
                    --seed 1\
                    --reid_cls_names macaque\
                    --val_intervals 20\
                    --save_all
                    
                    # --resume\
                    #--use_gc\
                    #--gc_cls_names 'Richard,Kiwi,Timo,Alex,Flo'\
                    # --cat_spec_wh\
                    
                    
                    #--trainval\
                    
                    #--train_only_gc '../models/model_120.pth'
cd ..
