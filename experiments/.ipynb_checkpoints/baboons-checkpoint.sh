cd src
python train.py mot --exp_id baboons --gpus 0 --batch_size 2 --load_model '../models/mcqcpz/model_200.pth' --num_epochs 200 --lr_step '150' --data_cfg '../src/lib/cfg/baboons.json'
cd ..
