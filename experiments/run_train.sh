cd src
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0,1,2,3
python train.py mot --exp_id test_cat-spec-wh_high-batch --gpus 0,1,2,3 --batch_size 24 --load_model '../models/model_120.pth' --num_epochs 60 --lr_step '20' --data_cfg '../src/lib/cfg/lemur_box.json' --reid_cls_names 'lemur,box' --reid_cls_ids '0,1' --cat_spec_wh #--trainval --val_intervals 2 #--use_pose #--train_only_pose '../models/model_120.pth'
cd ..
