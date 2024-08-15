#! /bin/sh

cd src
python train.py mot --exp_id macaquecp_hrnet_pretrained\
                    --load_model '../models/hrnetv2_w32_imagenet_pretrained.pth'\
                    --num_epochs 150\
                    --lr_step 100\
                    --lr '1e-4'\
                    --data_cfg '../src/lib/cfg/macaquecopypaste.json'\
                    --store_opt\
                    --arch hrnet_32\
                    --gpus 0,1,2,3\
                    --batch_size 8\
                    --seed 13\
                    --reid_cls_names macaque\
                    --val_intervals 10\
                    --save_all
cd ..


