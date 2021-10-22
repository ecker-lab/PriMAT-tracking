cd src
python train.py mot --exp_id macaquecpz --load_model '../models/ctdet_coco_dla_2x.pth' --num_epochs 250 --lr_step '200' --gpus 0,1,2,3 --batch_size 32 --data_cfg '../src/lib/cfg/macaquecopypaste.json' --data_dir '/local/eckerlab/' 
cd ..