import os

os.chdir('/home/matthias/monkey/MC-FairMOT_mt/src')

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "4" 
os.system("python train.py mot --exp_id lab_all_30-3 --gpus 0 --batch_size 6 --load_model '../models/model_120.pth' --num_epochs 140 --lr 1e-1 --lr_step '20,40,60,80,100,120' --input_h 768 --input_w 1024 --data_cfg '../src/lib/cfg/lab.json'")
