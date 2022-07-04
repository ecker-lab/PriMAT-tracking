cd src
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0
python train.py mot --exp_id testing_things --gpus 0 --batch_size 1 --load_model '../exp/mot/lab_all_30-3/model_last.pth' --num_epochs 100 --lr_step '20,60' --data_cfg '../src/lib/cfg/lab.json' --trainval --val_intervals 10 --train_only_pose #--cat_spec_wh '../models/model_120.pth'
cd ..
