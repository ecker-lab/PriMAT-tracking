#!/bin/bash
#SBATCH --gres=gpu:rtx5000:1
#SBATCH -p gpu
#SBATCH -t 0-05:00:00
#SBATCH -o /usr/users/vogg/monkey-tracking-in-the-wild/slurm_files/job-%J.out


cd /usr/users/vogg/monkey-tracking-in-the-wild/src
module load anaconda3
source activate /usr/users/vogg/.conda/envs/mktrack

# Randomly decide whether to include --squared_bboxes
#if [ $((RANDOM % 2)) -eq 0 ]; then
#    squared_bboxes_option="--squared_bboxes"
#    method="square"
#else
#    squared_bboxes_option=""
#    method="exact"
#fi

squared_bboxes_option=""
method="square"

move_px=20 #$((RANDOM % 101))

# Generate random values for zoom_min and zoom_max
#zoom_min=$(( RANDOM % 71 + 50 ))  # Random number between 50 and 120
zoom_min=100
zoom_max=150 #$(( RANDOM % 71 + 80 ))  # Random number between 80 and 150

# Ensure zoom_min is strictly smaller than zoom_max
#while [ $zoom_min -ge $zoom_max ]; do
    #zoom_min=$(( RANDOM % 71 + 50 ))
    #zoom_max=$(( RANDOM % 71 + 80 ))
#done

# Convert to floating-point values (dividing by 100)
zoom_min=$(echo "scale=2; $zoom_min / 100" | bc)
zoom_max=$(echo "scale=2; $zoom_max / 100" | bc)




python train.py mot --exp_id final_ids/alpha\
                    --gpus 0 \
                    --batch_size 1\
                    --load_model '../models/lemur_tracking_2500.pth'\
                    --num_epochs 100\
                    --lr_step 30\
                    --lr '1e-5'\
                    --data_cfg '../src/lib/cfg/lemur_ids_cleaned1.json'\
                    --store_opt\
                    --arch hrnet_32\
                    --data_dir '/usr/users/vogg/Labelling/Lemurs/Individual_imgs/'\
                    --seed 13\
                    --reid_cls_names lemur,box\
                    --use_gc\
                    --gc_cls_names Cha,Flo,Gen,Geo,Her,Rab,Red,Uns\
                    --gc_with_roi\
                    --save_all\
                    --val_intervals 2\
                    --gc_dim 3\
                    --train_only_gc\
                    $squared_bboxes_option\
                    --move_px $move_px\
                    --zoom_min $zoom_min\
                    --zoom_max $zoom_max

cd ..

