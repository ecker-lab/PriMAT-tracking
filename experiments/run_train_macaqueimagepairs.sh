#!/bin/bash
#SBATCH --gres=gpu:rtx5000:4
#SBATCH -p gpu
#SBATCH -t 2-00:00:00
#SBATCH -o /usr/users/vogg/monkey-tracking-in-the-wild/slurm_files/job-%J.out

cd /local/eckerlab/

if [ ! -d "MacaqueImagePairs" ] 
then
    tar xf /usr/users/agecker/datasets/MacaqueImagePairs.tar
fi
    
source activate mktrack

cd /usr/users/vogg/monkey-tracking-in-the-wild/src
#--load_model '../models/hrnetv2_w32_imagenet_pretrained.pth'\

python train.py mot --exp_id mcqimgpairs_nopretrain\
                    --num_epochs 500\
                    --lr_step 200\
                    --lr '5e-5'\
                    --data_cfg '../src/lib/cfg/macaque_image_pairs.json'\
                    --store_opt\
                    --arch hrnet_32\
                    --gpus 0,1,2,3\
                    --batch_size 8\
                    --data_dir '/local/eckerlab/'\
                    --seed 3\
                    --reid_cls_names 'macaque'\
                    --val_intervals 20\
                    --save_all
                    # --resume\
                    #--use_gc\
                    #--gc_cls_names 'Richard,Kiwi,Timo,Alex,Flo'\
                    # --cat_spec_wh\
                    #--trainval\
                    #--train_only_gc '../models/model_120.pth'
cd ..