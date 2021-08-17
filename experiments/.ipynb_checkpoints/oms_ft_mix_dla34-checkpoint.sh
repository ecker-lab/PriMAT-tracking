cd src
python train.py mot --exp_id oms_only --load_model '../models/oms/oms6.pth' --num_epochs 4 --lr_step '15' --gpus 0 --batch_size 4 --data_cfg '../src/lib/cfg/openmonkeystudio.json' --data_dir '/local/eckerlab/'
cd ..