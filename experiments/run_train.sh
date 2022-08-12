cd src
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0,1,2,3
python train.py mot --exp_id lemur_box --gpus 0,1,2,3 --batch_size 12 --load_model '../models/model_120.pth' --num_epochs 400 --lr_step '200' --data_cfg '../src/lib/cfg/lemur_box.json' #--use_pose #--trainval --val_intervals 10 --train_only_pose #--cat_spec_wh '../models/model_120.pth'
cd ..
