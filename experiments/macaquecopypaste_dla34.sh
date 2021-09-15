cd src
python train.py mot --exp_id macaquecpz --load_model '../models/mcqcpz/mcqcpz160.pth' --num_epochs 120 --lr_step '120' --gpus 0,1,2,3 --batch_size 32 --data_cfg '../src/lib/cfg/macaquecopypaste.json' --data_dir '/local/eckerlab/'
cd ..