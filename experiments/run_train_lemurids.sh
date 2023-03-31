cd src
python train.py mot --exp_id lemur_ids\
                    --gpus 0 --batch_size 1\
                    --load_model '../models/hrnet32_lemur_sep22.pth'\
                    --num_epochs 100\
                    --lr_step 40\
                    --lr '5e-5'\
                    --data_cfg '../src/lib/cfg/lemur_ids.json'\
                    --store_opt\
                    --arch hrnet_32\
                    --data_dir '/local/eckerlab/'\
                    --seed 13\
                    --reid_cls_names lemur,box\
                    --use_gc\
                    --gc_cls_names Cha,Flo,Gen,Geo,Her,Rab,Red,Uns\
                    --no_aug_hsv
                    # --cat_spec_wh\
                    # --resume\
                    #--trainval\
                    #--val_intervals 10\
                    #--save_all
                    #--train_only_gc '../models/model_120.pth'
cd ..


#Amb,Cam,Cha,Che,Flo,Gen,Geo,Har,Her,Isa,Kai,Lat,Mya,Pal,Rab,Red,Sap,Taj