cd src
python train.py mot --exp_id lemur_ids_head\
                    --gpus 0 --batch_size 1\
                    --load_model '../models/hrnet32_lemur_sep22.pth'\
                    --num_epochs 50\
                    --lr_step 30\
                    --lr '2e-5'\
                    --data_cfg '../src/lib/cfg/lemur_ids_cleaned1.json'\
                    --store_opt\
                    --arch hrnet_32\
                    --seed 2\
                    --reid_cls_names lemur,box\
                    --use_gc\
                    --gc_cls_names Cha,Flo,Gen,Geo,Her,Rab,Red,Uns\
                    --save_all\
                    --gc_dim 128\
                    --val_intervals 2\
                    --train_only_gc
                    #--no_aug_hsv\
                    # --cat_spec_wh\
                    # --resume\
                    #--trainval\
                    #--train_only_gc '../models/model_120.pth'
cd ..


#Amb,Cam,Cha,Che,Flo,Gen,Geo,Har,Her,Isa,Kai,Lat,Mya,Pal,Rab,Red,Sap,Taj