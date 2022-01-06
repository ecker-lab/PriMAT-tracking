cd src
python train.py mot --exp_id unnecessary --load_model '../models/fairmot_dla34.pth' --num_epochs 500 --lr_step '250' --gpus 0,1,2,3 --batch_size 8 --data_cfg '../src/lib/cfg/macaquepose.json' --data_dir '/local/eckerlab/'
cd ..