#! /bin/sh

cd /usr/users/vogg/monkey-tracking-in-the-wild/src




for vid in alpha_ind2 alpha_ind3 # 

do
    for i in 24 #10 20 30 40 50 60 70 80 90 100 110 120 130 140 150 160 170 180 190 200 210 220 230 240 250
    do

    
#--output_root ../videos/lemurs/"$vid"/buffer_"$iou"/\ 

python demo.py mot  --load_model ../exp/mot/lemur_ids_roi/model_36.pth\
                    --conf_thres 0.04\
                    --det_thres 0.6\
                    --new_overlap_thres 0.85\
                    --sim_thres 0.8\
                    --input_video /usr/users/agecker/datasets/lemur_videos_eval/Videos/"$vid".mp4\
                    --output_root ../videos/lemur_ids/"$vid"/\
                    --output_name "$vid"\
                    --store_opt\
                    --line_thickness 2\
                    --arch hrnet_32\
                    --output_format video\
                    --reid_cls_names "lemur,box"\
                    --use_gc\
                    --gc_cls_names Cha,Flo,Gen,Geo,Her,Rab,Red,Uns\
                    --proportion_iou 0.2\
                    --double_kalman\
                    --gc_with_roi\
                    --gc_dim 3\
                    #--debug_info\
                    #--seed 42
                    #--min-box-area 100
                    #--use_gc\
                    # --cat_spec_wh\
                    # --input_h 768\
                    # --input_w 1024\
                    # --id_inline\
done
done
cd ..