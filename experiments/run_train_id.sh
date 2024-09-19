#!/bin/bash

cd src

python train_id.py id --exp_id lemurs_full\
                    --id_train_root '../../Labelling/Lemurs/Individual_imgs/'\
                    --id_train_file '../src/data/lemur_ids_base.train'\
                    --id_val_root 'path/to/val_images'\
                    --id_val_file '../src/data/lemur_ids_base.val'\
                    --output_path "../exp/id/batch_id_training/"\
                    --num_epochs 200\
                    --store_opt\
                    --batch_size 2\
                    --seed 2\
                    --val_intervals 50\
                    --save_all\
                    --use_gc\
                    --gc_cls_names "Cha,Flo,Gen,Geo,Her,Rab,Red,Uns"\
                    --reid_cls_names "lemur,box"\
                    --cat_spec_wh\
                    --K 50\
                    --down_ratio 4\
                    --ltrb\
                    --mse_loss\
                    --gc_with_roi\
                    --hm_gauss\
                    --no_aug_hsv
cd ..
