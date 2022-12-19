cd src
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=2,3
python train.py mot --exp_id without_WH-lemur_box-7-11-22 --gpus 0,1 --batch_size 12 --load_model '../models/model_120.pth' --num_epochs 60 --lr_step '20,40' --data_cfg '../src/lib/cfg/lemur_box.json' --store_opt --reid_cls_names 'lemur,box' --reid_cls_ids '0,1' #--cat_spec_wh --use_pose --trainval --val_intervals 2 #--train_only_pose '../models/model_120.pth'
cd ..
