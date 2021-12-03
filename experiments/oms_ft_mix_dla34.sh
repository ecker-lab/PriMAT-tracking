cd src
python train.py mot --exp_id omszero --load_model '../models/ctdet_coco_dla_2x.pth' --num_epochs 10 --lr_step '15' --gpus 0,1 --batch_size 8 --data_cfg '../src/lib/cfg/openmonkeystudio.json' --data_dir '/local/eckerlab/'
cd ..