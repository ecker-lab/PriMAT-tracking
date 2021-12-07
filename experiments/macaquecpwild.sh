cd src
python train.py mot --exp_id macaquecpwild --load_model '../models/ctdet_coco_dla_2x.pth' --num_epochs 250 --lr_step '250' --lr '1e-4' --gpus 0,1,2,3 --batch_size 32 --data_cfg '../src/lib/cfg/macaquecpwild.json' --data_dir '/local/eckerlab/' 
cd ..