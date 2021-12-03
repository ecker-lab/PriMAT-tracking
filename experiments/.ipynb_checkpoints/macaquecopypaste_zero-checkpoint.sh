cd src
python train.py mot --exp_id macaquecpz1e3 --load_model '../models/ctdet_coco_dla_2x.pth' --num_epochs 300 --lr_step '250' --lr '1e-3' --gpus 0,1,2,3 --batch_size 32 --data_cfg '../src/lib/cfg/macaquecopypaste.json' --data_dir '/local/eckerlab/' 
cd ..