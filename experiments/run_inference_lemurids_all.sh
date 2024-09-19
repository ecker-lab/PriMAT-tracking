#! /bin/sh


cd /usr/users/vogg/monkey-tracking-in-the-wild/src



#for vid in A_e7_c1 A_e7_c2 A_e7_c3 A_e7_c4 A_e8_c1 A_e8_c2 A_e8_c3 A_e8_c4 #A_e1_c1 A_e1_c2 A_e1_c3 A_e1_c4 A_e2_c1 A_e2_c2 A_e2_c3 A_e2_c4 A_e3_c1 A_e3_c2 A_e3_c3 A_e3_c4 A_e4_c1 A_e4_c2 A_e4_c3 A_e4_c4 A_e5_c1 A_e5_c2 A_e5_c3 A_e5_c4
for vid in a_e_2_220919_c1 a_e_2_220919_c2 a_e_2_220919_c3 a_e_2_220919_c4 a_e_3_220920_c1 a_e_3_220920_c2 a_e_3_220920_c3 a_e_4_220921_c3 a_e_5_220927_c3 a_e_6_220928_c4 a_e_8_221005_c4 a_e_9_221007_c4 a_e_10_221008_c4 #a_e_4_220921_c2 a_e_5_220927_c1 a_e_5_220927_c2 a_e_7_221004_c3 #a_e_10_221008_c1 a_e_10_221008_c2 #a_e_7_221004_c1 a_e_8_221005_c1 a_e_8_221005_c2 a_e_9_221007_c1 a_e_9_221007_c2 #a_e_1_220918_c1 a_e_1_220918_c2 a_e_1_220918_c3 a_e_6_220928_c1 a_e_6_220928_c2

do
    for i in 22 #30 40 50 #80 100 #10 20 30 40 50 60 70 80 90 100 110 120 130 140 150 160 170 180 190 200 210 220 230 240 250
    do

    for model in batch_cleaned1
    do

    
# #../exp/mot/lemur_ids_additional/model_20.pth\
#/usr/users/agecker/datasets/lemur_ids/alpha_ID_validation_snippets/"$vid".mp4\
#../videos/lemur_id_validation/basedata/"$vid"/\ 
#--input_video /usr/users/agecker/datasets/lemur_ids/alpha_ID_validation_snippets/"$vid".mp4\
#--load_model ../exp/mot/final_ids/"$model"/model_"$i".pth\
#/usr/users/vogg/sfb1528s3/B06/2023april-july/NewBoxesClosed/Converted/Alpha/"$vid".mp4\
# "../exp/id/batch_id_training/cleaned1/model_checkpoint_22.pth"\

python demo.py mot  --load_tracking_model "../exp/mot/lemurs_full/model_100.pth"\
                    --load_id_model ../models/georg_model1/large_scale_model_2.pth\
                    --conf_thres 0.02\
                    --det_thres 0.6\
                    --new_overlap_thres 0.85\
                    --sim_thres 0.8\
                    --input_video /usr/users/agecker/datasets/lemur_experiments_sep22/Converted/alpha/"$vid".mp4\
                    --output_root ../videos/lemur_ids_georg/sep22/alpha/"$vid"/\
                    --output_name "$vid"\
                    --store_opt\
                    --line_thickness 2\
                    --arch hrnet_32\
                    --output_format text\
                    --reid_cls_names lemur,box\
                    --gc_cls_names Cha,Flo,Gen,Geo,Her,Rab,Red,Uns\
                    --proportion_iou 0.2\
                    --double_kalman\
                    --gc_dim 3\
                    --gc_with_roi\
                    --use_gc
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
done
cd ..