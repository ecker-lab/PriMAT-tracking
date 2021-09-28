cd src
python train.py mot --exp_id macaquecp32 --load_model '../models/mcqcp32/mcqcp32_160.pth' --num_epochs 140 --lr_step '120' --gpus 0,1,2,3 --batch_size 32 --data_cfg '../src/lib/cfg/macaquecopypaste.json' --data_dir '/local/eckerlab/' --reid_dim 32
cd ..