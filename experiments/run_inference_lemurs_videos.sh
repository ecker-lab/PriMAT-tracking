#! /bin/sh

cd /usr/users/vogg/monkey-tracking-in-the-wild/src




for vid in Eval8 Eval12 Eval16

do
    for i in 250 #10 20 30 40 50 60 70 80 90 100 110 120 130 140 150 160 170 180 190 200 210 220 230 240 250
    do

    out=$((i * 2))
    
    
#--load_model ../exp/mot/lemurs/model_"$out".pth\
python demo.py mot  --load_model ../exp/mot/lemur_1500/model_400.pth\
                    --conf_thres 0.04\
                    --det_thres 0.5\
                    --new_overlap_thres 0.8\
                    --sim_thres 0.8\
                    --input_video /usr/users/agecker/datasets/lemur_videos_eval/Videos/"$vid".mp4\
                    --output_root ../videos/lemurs/video_dump/"$vid"/\
                    --output_name "$vid"\
                    --store_opt\
                    --line_thickness 2\
                    --debug_info\
                    --arch hrnet_32\
                    --output_format video\
                    --reid_cls_names "lemur,box"\
                    --proportion_iou 0.5\
                    --buffered_iou 0\
                    --double_kalman
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
cd ..