cd /usr/users/vogg/monkey-tracking-in-the-wild/src

#--load_model '../models/hrnetv2_w32_imagenet_pretrained.pth'\
python train.py mot --exp_id lemurs_nopretrain\
                    --num_epochs 500\
                    --lr_step 300\
                    --lr '1e-4'\
                    --data_cfg '../src/lib/cfg/lemur_box.json'\
                    --store_opt\
                    --arch hrnet_32\
                    --gpus 0\
                    --batch_size 2\
                    --data_dir '/usr/users/vogg/Labelling/Lemurs/'\
                    --seed 2\
                    --reid_cls_names lemur,box\
                    --val_intervals 20\
                    --save_all
                    # --resume\
                    #--use_gc\
                    #--gc_cls_names 'Richard,Kiwi,Timo,Alex,Flo'\
                    # --cat_spec_wh\
                    
                    
                    #--trainval\
                    
                    #--train_only_gc '../models/model_120.pth'
cd ..
