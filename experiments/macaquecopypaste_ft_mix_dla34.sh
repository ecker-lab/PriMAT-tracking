cd src
python train.py mot --exp_id macaquecopypaste --load_model '../models/mcqcp/mcqcp180.pth' --num_epochs 40 --lr_step '40' --gpus 0 --batch_size 4 --data_cfg '../src/lib/cfg/macaquecopypaste.json' --data_dir '/local/eckerlab/'
cd ..