cd src
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0,1,2
python train.py mot --exp_id testing_fixes_gc\
                    --gpus 0,1,2 --batch_size 18\
                    --load_model '../models/model_120.pth'\
                    --num_epochs 60\
                    --lr_step '20,40'\
                    --data_cfg '../src/lib/cfg/lab.json'\
                    --store_opt\
                    --use_gc\
                    --gc_cls_names 'Richard,Kiwi,Timo,Alex,Flo'\
                    # --cat_spec_wh\
                    # --seed 13
                    # --reid_cls_names 'lemur,box'\
                    
                    # --resume\
                    
                    
                    #--trainval\
                    #--val_intervals 20\
                    #--train_only_gc '../models/model_120.pth'
cd ..
