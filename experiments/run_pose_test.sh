cd src
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0,1,2,3
python test_pose.py mot --exp_id eval_exp_room --gpus 0,1,2,3 --batch_size 1 --load_model '../models/mcFairmotPose_without-val.pth' --data_cfg '../src/lib/cfg/explorationroom_largest.json' --use_pose #--trainval --val_intervals 10 --train_only_pose #--cat_spec_wh '../models/model_120.pth'
cd ..
