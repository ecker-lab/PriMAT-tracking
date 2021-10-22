cd src
python train.py mot --exp_id macaquecp --load_model '../models/fairmot_dla34.pth' --num_epochs 250 --lr_step '200' --gpus 0,1,2,3 --batch_size 32 --data_cfg '../src/lib/cfg/macaquecopypaste.json' --data_dir '/local/eckerlab/' 
cd ..