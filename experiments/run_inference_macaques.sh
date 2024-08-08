#! /bin/sh
#SBATCH --gres=gpu:RTX5000:1
#SBATCH -p gpu
#SBATCH -t 2-00:00:00
#SBATCH -o /usr/users/vogg/monkey-tracking-in-the-wild/slurm_files/job-%J.out


module load anaconda3
source activate /usr/users/vogg/.conda/envs/mktrack


cd /usr/users/vogg/monkey-tracking-in-the-wild/src



for i in 200 #10 20 30 40 50 60 70 80 90 100 110 120 130 140 150 160 170 180 190 200
do
for conf in 0.04 #0.01 0.02 0.04 0.1 0.2 0.4 #0.06 #0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1
    do
    for vid in vid_630 vid_038 vid_130 vid_251 vid_307 vid_440 vid_455 vid_722 vid_817 vid_846 vid_854 vid_942
    do
    

    out=$((i * 2))

    for det in 0.5 #0.4 0.5 0.6
    do
    for sim in 0.8 #0.7 0.8 0.9
    do
    for seed in 1 #2 3
    do
    for model in macaquecp #imagenet macaquecpw nopretrain #
    do

    for prop in 0.8 #0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1
    do
    
#--output_root ../videos/lemurs/"$vid"/buffer_"$iou"/\ 
#../exp/mot/macaquecpw_seed2/model_"$out".pth\
#../videos/methods_paper_ablations/macaquesw_"$conf"_"$det"_"$sim"_"$prop"/singlekalman/\
#../videos/methods_paper_ablations2/macaques_"$conf"_"$det"_"$new"/singlekalman/\
#../videos/methods_paper_3seeds_epoch200/macaques_"$model"_"$seed"\#
python demo.py mot  --load_tracking_model ../exp/mot/paper/macaques_"$model"_"$seed"_5e-5/model_"$out".pth\
                    --conf_thres "$conf"\
                    --det_thres "$det"\
                    --new_overlap_thres 0.8\
                    --sim_thres "$sim"\
                    --input_video /usr/users/agecker/datasets/macaque_videos_eval/Videos/"$vid".mp4\
                    --output_root ../videos/video_dump/"$vid"\
                    --output_name "$vid"\
                    --store_opt\
                    --line_thickness 2\
                    --debug_info\
                    --arch hrnet_32\
                    --output_format video\
                    --reid_cls_names "macaque"\
                    --proportion_iou "$prop"
                    #--double_kalman
done
done
done
done
done
done
done
done
cd ..