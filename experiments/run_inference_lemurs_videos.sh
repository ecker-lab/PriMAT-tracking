#! /bin/sh
#SBATCH --gres=gpu:RTX5000:1
#SBATCH -p gpu
#SBATCH -t 0-12:00:00
#SBATCH -o /usr/users/vogg/monkey-tracking-in-the-wild/slurm_files/job-%J.out


module load anaconda3
#module load ffmpeg/6.0 
source activate /usr/users/vogg/.conda/envs/mktrack



cd /usr/users/vogg/monkey-tracking-in-the-wild/src



for i in 200 #10 20 30 40 50 60 70 80 90 100 110 120 130 140 150 160 170 180 190 200
do
for conf in 0.01 #0.01 0.02 0.04 0.1 0.2 0.4 #0.06 #0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1
    do
    for vid in Eval16 Eval17 Eval18 Eval19 Eval8 Eval9 Eval10 Eval11 Eval12 Eval13 Eval14 Eval15
    do
    

    out=$((i * 2))

    for det in 0.5 #0.4 0.5 0.6
    do
    for sim in 0.8 #0.7 0.8 0.9
    do
    for seed in 1 #2 3
    do
    for model in imagenet #macaquecp macaquecpw nopretrain
    do

    for prop in 0.8 #0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1
    do
    
#/usr/users/agecker/datasets/lemurid_validation/"$vid".mp4\
#../exp/mot/paper/lemur_"$model"_1_"$lr"/model_"$out".pth\
#../videos/methods_paper/lemurs_"$model"_"$i"_"$lr"\
# ../videos/methods_paper_ablations/lemurs_"$conf"_"$det"_"$sim"_"$prop"/singlekalman/\
#--output_root ../videos/methods_paper_ablations/lemursw_"$conf"_"$det"_"$sim"_"$prop"/doublekalman/\
#../videos/methods_paper_3seeds_epoch200/lemurs_"$model"_"$seed"/\
python demo.py mot  --load_tracking_model ../exp/mot/paper/lemur_"$model"_"$seed"_5e-5/model_"$out".pth\
                    --conf_thres "$conf"\
                    --det_thres "$det"\
                    --new_overlap_thres 0.8\
                    --sim_thres "$sim"\
                    --input_video /usr/users/agecker/datasets/lemur_videos_eval/Videos/"$vid".mp4\
                    --output_root ../videos/video_dump/"$vid"\
                    --output_name "$vid"\
                    --store_opt\
                    --line_thickness 2\
                    --debug_info\
                    --arch hrnet_32\
                    --output_format video\
                    --reid_cls_names "lemur,box"\
                    --proportion_iou "$prop"\
                    --double_kalman
                    #--buffered_iou 0\
                    #--seed 42
                    #--min-box-area 100
                    #--use_gc\
                    #--gc_cls_names 'Richard,Kiwi,Timo,Alex,Flo'\
                    # --cat_spec_wh\
                    # --input_h 768\
                    # --input_w 1024\
                    # --id_inline\
done
done
done
done
done
done
done
done
cd ..