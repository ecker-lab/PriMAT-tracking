cd src
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=2,3
python train.py mot --exp_id lab_all --gpus 0,1 --batch_size 4 --load_model '../models/model_120.pth' --num_epochs 100 --lr_step '20' --data_cfg '../src/lib/cfg/lab.json'
cd ..
