cd src
python train.py mot --exp_id macaquepose --load_model '../models/mix/mcq30_oms2.pth' --num_epochs 90 --lr_step '15' --gpus 0 --batch_size 4 --data_cfg '../src/lib/cfg/macaquepose.json' --data_dir '/local/eckerlab/'
cd ..