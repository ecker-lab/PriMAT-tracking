#! /bin/sh

cd src


for vid in vid_630 vid_038 vid_130 vid_251 vid_307 vid_440 vid_455 vid_722 vid_817 vid_846 vid_854 vid_942

do
    for i in 80 #10 20 30 40 50 60 70 80 90 100 110 120 130 140 150 160 170 180 190 200 210 220 230 240 250
    do


python demo.py mot  --load_model ../exp/mot/macaquecpw_seed2/model_"$i".pth\
                    --conf_thres 0.02\
                    --det_thres 0.5\
                    --new_overlap_thres 0.8\
                    --sim_thres 0.8\
                    --input_video ../macaque_videos_eval/Videos/"$vid".mp4\
                    --output_root ../videos/methods_paper/"$vid"/macaques_nopretrain_"$i"/\
                    --output_name "$vid"\
                    --store_opt\
                    --line_thickness 2\
                    --debug_info\
                    --arch hrnet_32\
                    --output_format text\
                    --reid_cls_names "macaque"\
                    --proportion_iou 0.2\
                    --double_kalman

done
done
cd ..