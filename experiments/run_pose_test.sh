cd src
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=3
python test_pose.py mot --exp_id eval_exp_room_pose-new --gpus 0 --batch_size 1 --load_model '../exp/mot/test_pose_high_batch-loss_fix3/model_last.pth' --data_cfg '../src/lib/cfg/explorationroom_largest.json' --use_pose #--trainval --val_intervals 10 --train_only_pose #--cat_spec_wh '../models/model_120.pth'
cd ..
