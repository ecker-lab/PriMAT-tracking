cd src
python train.py mot --exp_id wildmacaques --load_model '../models/mcqcp/mcqcp200.pth' --num_epochs 200 --lr_step '150' --gpus 0 --batch_size 4 --data_cfg '../src/lib/cfg/wildmacaques.json' --data_dir '/usr/users/agecker/datasets/'
cd ..