#!/bin/bash
#SBATCH --gres=gpu:rtx5000:4
#SBATCH --qos=long
#SBATCH -p gpu
#SBATCH -t 5-00:00:00
#SBATCH -o /usr/users/vogg/monkey-tracking-in-the-wild/slurm_files/job-%J.out

cd /local/eckerlab/

if [ ! -d "MacaquePose" ] 
then
    tar xf /usr/users/agecker/datasets/MacaquePose.tar
fi

if [ ! -d "MacaqueCopyPasteWild" ] 
then
    tar xf /usr/users/agecker/datasets/MacaqueCopyPasteWild.tar
fi

source activate mktrack

cd /usr/users/vogg/monkey-tracking-in-the-wild/src
python train.py mot --exp_id macaquecpw_seed3\
                    --load_model '../models/hrnetv2_w32_imagenet_pretrained.pth'\
                    --num_epochs 250\
                    --lr_step 100\
                    --lr '1e-4'\
                    --data_cfg '../src/lib/cfg/macaquecpwild.json'\
                    --store_opt\
                    --arch hrnet_32\
                    --gpus 0,1,2,3\
                    --batch_size 8\
                    --data_dir '/local/eckerlab/'\
                    --seed 3\
                    --reid_cls_names macaque\
                    --val_intervals 10\
                    --save_all
                    
                    # --resume\
                    #--use_gc\
                    #--gc_cls_names 'Richard,Kiwi,Timo,Alex,Flo'\
                    # --cat_spec_wh\
                    
                    
                    #--trainval\
                    
                    #--train_only_gc '../models/model_120.pth'
cd ..
