cd src
python train.py mot --exp_id macaqueposezero1e4 --load_model '../models/ctdet_coco_dla_2x.pth' --num_epochs 500 --lr_step '500' --lr '1e-4' --gpus 0,1,2,3 --batch_size 32 --data_cfg '../src/lib/cfg/macaquepose.json' --data_dir '/local/eckerlab/'
cd ..