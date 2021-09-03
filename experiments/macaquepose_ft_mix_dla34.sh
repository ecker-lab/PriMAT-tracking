cd src
python train.py mot --exp_id macaquepose --load_model '../models/mcqpose/mcq220.pth' --num_epochs 80 --lr_step '40' --gpus 0 --batch_size 4 --data_cfg '../src/lib/cfg/macaquepose.json' --data_dir '/local/eckerlab/'
cd ..
