cd src
python train.py mot --exp_id crowdhuman_dla34 --gpus 0 --batch_size 1 --load_model '../models/ctdet_coco_dla_2x.pth' --num_epochs 1 --lr_step '50' --data_cfg '../src/lib/cfg/crowdhuman.json'
cd ..