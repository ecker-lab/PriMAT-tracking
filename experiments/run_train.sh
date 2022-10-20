cd src
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0,1,2,3
python train.py mot --exp_id WH-Pose-17-10-22 --gpus 0,1,2,3 --batch_size 24 --load_model '../models/model_120.pth' --num_epochs 60 --lr_step '20,40' --data_cfg '../src/lib/cfg/explorationroom_largest.json' --use_pose --cat_spec_wh --store_opt #--reid_cls_names 'lemur,box' --reid_cls_ids '0,1' #--trainval --val_intervals 2 #--train_only_pose '../models/model_120.pth'
cd ..
