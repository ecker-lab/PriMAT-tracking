#! /bin/sh

cd /usr/users/vogg/monkey-tracking-in-the-wild/src




for vid in a_e_1_220918_c1 a_e_1_220918_c2 a_e_1_220918_c3 a_e_1_220918_c4 a_e_2_220919_c1 a_e_2_220919_c2 a_e_2_220919_c3 a_e_2_220919_c4 #e9_c3_3_5220_6960 e5_c1_1_1740_3480 e5_c3_4_6960_8700 e5_c4_1_1740_3480 e6_c3_3_5220_6960 e6_c4_4_6960_8700 e7_c3_6_10440_12180 e7_c3_14_24360_26100 e7_c4_9_15660_17400 e8_c3_2_3480_5220 e8_c4_2_3480_5220 e10_c3_0_0_1740 e10_c3_3_5220_6960

do
    for i in 250 #10 20 30 40 50 60 70 80 90 100 110 120 130 140 150 160 170 180 190 200 210 220 230 240 250
    do

    out=$((i * 2))
    
    
#--load_model '../models/hrnet32_lemur_sep22.pth'\
#/usr/users/agecker/datasets/lemurid_validation/"$vid".mp4\
python demo.py mot  --load_model ../exp/mot/lemurs_close/model_181.pth\
                    --conf_thres 0.01\
                    --det_thres 0.5\
                    --new_overlap_thres 0.8\
                    --sim_thres 0.8\
                    --input_video /usr/users/agecker/datasets/lemur_experiments_sep22/Converted/alpha/"$vid".mp4
                    --output_root ../videos/lemurs/"$vid"/\
                    --output_name "$vid"\
                    --store_opt\
                    --line_thickness 2\
                    --debug_info\
                    --arch hrnet_32\
                    --output_format video\
                    --reid_cls_names "lemur,box"\
                    --proportion_iou 0\
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
cd ..