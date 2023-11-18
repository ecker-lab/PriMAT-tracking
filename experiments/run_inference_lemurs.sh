#! /bin/sh

cd /usr/users/vogg/monkey-tracking-in-the-wild/src




for vid in a_e_1_220918_c1 a_e_1_220918_c2 a_e_1_220918_c3 a_e_1_220918_c4 a_e_1_220918_c6 a_e_1_220918_c7 a_e_1_220918_c8 a_e_1_220918_c9 #Eval16 Eval17 Eval18 Eval19 Eval8 Eval9 Eval10 Eval11 Eval12 Eval13 Eval14 Eval15

do
    for i in 250 #10 20 30 40 50 60 70 80 90 100 110 120 130 140 150 160 170 180 190 200 210 220 230 240 250
    do

    for prop in 0.5 #0 0.2 0.4 0.6 0.8 1
    do
    
    out=$((i * 2))
    
#--output_root ../videos/lemurs/"$vid"/buffer_"$iou"/\ 

python demo.py mot  --load_model '../models/hrnet32_lemur_sep22.pth'\
                    --conf_thres 0.04\
                    --det_thres 0.5\
                    --new_overlap_thres 0.8\
                    --sim_thres 0.8\
                    --input_video /usr/users/agecker/datasets/lemur_experiments_sep22/Converted/alpha/"$vid".mp4\
                    --output_root ../videos/lemur_tracking_output/"$vid"/\
                    --output_name "$vid"\
                    --store_opt\
                    --line_thickness 2\
                    --debug_info\
                    --arch hrnet_32\
                    --output_format text\
                    --reid_cls_names "lemur,box"\
                    --proportion_iou "$prop"\
                    --double_kalman
                    #--use_buffered_iou\
                    #--buffered_iou "$iou" \
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