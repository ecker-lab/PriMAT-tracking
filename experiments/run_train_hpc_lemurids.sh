#!/bin/bash
#SBATCH --gres=gpu:rtx5000:4
#SBATCH -p gpu
#SBATCH -t 2-00:00:00
#SBATCH -o /usr/users/vogg/monkey-tracking-in-the-wild/slurm_files/job-%J.out


cd /local/eckerlab/


if [ ! -d "Individual" ] 
then
    tar xf /usr/users/vogg/Labelling/Lemurs/Individual_imgs.tar
fi

source activate mktrack

cd /usr/users/vogg/monkey-tracking-in-the-wild/src


for lr_step in 80 #40 80 120
do

for lr in '1e-5' #'1e-5' '2e-5' '5e-5' '1e-4' '2e-4'
do

python train.py mot --exp_id lemur_ids_"$lr_step"_"$lr"\
                    --gpus 0 --batch_size 1\
                    --load_model '../models/hrnet32_lemur_sep22.pth'\
                    --num_epochs 200\
                    --lr_step "$lr_step"\
                    --lr "$lr"\
                    --data_cfg '../src/lib/cfg/lemur_ids_hpc.json'\
                    --store_opt\
                    --arch hrnet_32\
                    --seed 13\
                    --reid_cls_names lemur,box\
                    --use_gc\
                    --gc_cls_names Cha,Flo,Gen,Geo,Her,Rab,Red,Uns\
                    --save_all\
                    --val_intervals 10
                    #--train_only_gc
                    #--no_aug_hsv\
                    # --cat_spec_wh\
                    # --resume\
                    #--trainval\
                    #--train_only_gc '../models/model_120.pth'

done
done

cd ..


#Amb,Cam,Cha,Che,Flo,Gen,Geo,Har,Her,Isa,Kai,Lat,Mya,Pal,Rab,Red,Sap,Taj