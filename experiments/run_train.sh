cd src
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0
python train.py mot --exp_id test_changes2 --gpus 0 --batch_size 1 --load_model '../models/model_120.pth' --num_epochs 40 --lr_step '20' --data_cfg '../src/lib/cfg/lab.json' --trainval --val_intervals 2 #--use_pose #--train_only_pose #--cat_spec_wh '../models/model_120.pth'
cd ..
