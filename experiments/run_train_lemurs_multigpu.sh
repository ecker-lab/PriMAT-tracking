#!/bin/bash
#SBATCH --gres=gpu:RTX5000:4
#SBATCH -p gpu
#SBATCH -t 0-16:00:00
#SBATCH -o /usr/users/vogg/monkey-tracking-in-the-wild/slurm_files/job-%J.out

module load cuda/11.1.0
module load anaconda3
cd /local/eckerlab/



if [ ! -d "LemurLabellingNov22" ] 
then
    tar xf /usr/users/agecker/datasets/LemurLabellingNov22.tar
fi

source activate mktrack

cd /usr/users/vogg/monkey-tracking-in-the-wild/src

for seed in 3
do

for lr in '5e-5'
do
#'../exp/mot/macaquecp_seed1/model_200.pth'\ macaquecp pretraining
#'../models/hrnetv2_w32_imagenet_pretrained.pth'\

python train.py mot --exp_id paper/lemur_macaquecpw_"$seed"_"$lr"\
                    --load_tracking_model '../exp/mot/macaquecpw_seed1/model_200.pth'\
                    --num_epochs 400\
                    --lr_step 200\
                    --lr "$lr"\
                    --data_cfg '../src/lib/cfg/lemur_box.json'\
                    --store_opt\
                    --arch hrnet_32\
                    --gpus 0,1,2,3\
                    --batch_size 8\
                    --data_dir '/local/eckerlab/'\
                    --seed "$seed"\
                    --reid_cls_names 'lemur,box'\
                    --val_intervals 20\
                    --save_all
                    # --resume\
                    #--use_gc\
                    #--gc_cls_names 'Richard,Kiwi,Timo,Alex,Flo'\
                    # --cat_spec_wh\
                    #--trainval\
                    #--train_only_gc '../models/model_120.pth'

done
done
cd ..
