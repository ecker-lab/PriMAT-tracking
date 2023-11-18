#! /bin/sh

cd /usr/users/vogg/monkey-tracking-in-the-wild/src




for vid in vid_630 vid_038 vid_130 vid_251 vid_307 vid_440 vid_455 vid_722 vid_817 vid_846 vid_854 vid_942

do
    for i in 80 #10 20 30 40 50 60 70 80 90 100 110 120 130 140 150 160 170 180 190 200 210 220 230 240 250
    do

    for prop in 0.5 #0.2 0.4 0.6 0.8 1
    do
    
    out=$((i * 2))
    
#--output_root ../videos/lemurs/"$vid"/buffer_"$iou"/\ 

python demo.py mot  --load_model ../exp/mot/macaquecpw_seed2/model_"$out".pth\
                    --conf_thres 0.02\
                    --det_thres 0.5\
                    --new_overlap_thres 0.8\
                    --sim_thres 0.8\
                    --input_video /usr/users/agecker/datasets/macaque_videos_eval/Videos/"$vid".mp4\
                    --output_root ../videos/methods_paper/"$vid"/macaques_nopretrain_"$i"/\
                    --output_name "$vid"\
                    --store_opt\
                    --line_thickness 2\
                    --debug_info\
                    --arch hrnet_32\
                    --output_format text\
                    --reid_cls_names "macaque"\
                    --proportion_iou "$prop"\
                    --double_kalman
                    #--use_buffered_iou\
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
cd ..