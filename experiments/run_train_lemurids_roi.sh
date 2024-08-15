cd src

python train.py mot --exp_id 'lemur_ids_cleaned1_squareroi_jitter_move_weightedids'\
                    --gpus 0 \
                    --batch_size 1\
                    --load_model '../models/hrnet32_lemur_sep22.pth'\
                    --num_epochs 50\
                    --lr_step 30\
                    --lr '1e-5'\
                    --data_cfg '../src/lib/cfg/lemur_ids_cleaned1.json'\
                    --store_opt\
                    --arch hrnet_32\
                    --data_dir '../../Labelling/Lemurs/Individual_imgs/'\
                    --seed 13\
                    --reid_cls_names lemur,box\
                    --use_gc\
                    --gc_cls_names Cha,Flo,Gen,Geo,Her,Rab,Red,Uns\
                    --gc_with_roi\
                    --save_all\
                    --val_intervals 2\
                    --gc_dim 3\
                    --train_only_gc

cd ..

