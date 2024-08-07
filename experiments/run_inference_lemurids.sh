#! /bin/sh
#SBATCH --gres=gpu:rtx5000:1
#SBATCH -p gpu
#SBATCH -t 2-00:00:00
#SBATCH -o /usr/users/vogg/monkey-tracking-in-the-wild/slurm_files/job-%J.out

cd /usr/users/vogg/monkey-tracking-in-the-wild/src

module load anaconda3
source activate /usr/users/vogg/.conda/envs/mktrack

for vid in e7_c4_0_0_1740 e3_c4_8_13920_15660 e6_c3_2_3480_5220 e6_c3_0_0_1740 e6_c3_3_5220_6960 e1_c4_11_19140_20880 e1_c4_14_24360_26100 e1_c4_4_6960_8700 e3_c4_17_29580_31320 e3_c4_8_13920_15660 e3_c4_3_5220_6960 e4_c4_0_0_1740 e4_c4_3_5220_6960 e4_c4_9_15660_17400 e5_c4_5_8700_10440 e5_c4_6_10440_12180 e5_c4_7_12180_13920 e7_c2_10_17400_19140 e7_c2_4_6960_8700 e7_c2_8_13920_15660 e7_c4_0_0_1740 e7_c4_10_17400_19140 e7_c4_8_13920_15660 e8_c3_4_6960_8700 e8_c3_5_8700_10440 e8_c3_7_12180_13920 e9_c3_10_17400_19140 e9_c3_5_8700_10440 e9_c3_9_15660_17400 e10_c3_1_1740_3480 e10_c3_8_13920_15660 e10_c3_9_15660_17400
#a_e_3_220920_c1 #a_e_3_220920_c2 a_e_3_220920_c3 a_e_3_220920_c4 a_e_4_220921_c1 a_e_4_220921_c2 a_e_4_220921_c3 a_e_4_220921_c4 a_e_5_220927_c1 a_e_5_220927_c2 a_e_5_220927_c3 a_e_5_220927_c4 #e1_c7_4_6960_8700 e2_c6_10_17400_19140 #e9_c3_3_5220_6960 #e5_c1_1_1740_3480 #e5_c3_4_6960_8700 e5_c4_1_1740_3480 e6_c3_3_5220_6960 e6_c4_4_6960_8700 e7_c3_6_10440_12180 e7_c3_14_24360_26100 e7_c4_9_15660_17400 e8_c3_2_3480_5220 e8_c4_2_3480_5220 e10_c3_0_0_1740 e10_c3_3_5220_6960

do
    for i in 32 #30 40 50 #80 100 #10 20 30 40 50 60 70 80 90 100 110 120 130 140 150 160 170 180 190 200 210 220 230 240 250
    do

    for model in cleaned1
    do

    
#--output_root ../videos/lemurs/"$vid"/buffer_"$iou"/\ 
# #../exp/mot/lemur_ids_additional/model_20.pth\
#/usr/users/agecker/datasets/lemur_ids/alpha_ID_validation_snippets/"$vid".mp4\
#../videos/lemur_id_validation/basedata/"$vid"/\ 
#additional 22
#base 12
#cleaned 28
#../models/hrnet32_lemur_sep22.pth\

python demo.py mot  --load_tracking_model ../exp/mot/lemur_ids_head/model_"$i".pth\
                    --load_id_model ../exp/id/batch_id_training/"$model"/model_checkpoint_"$i".pth\
                    --conf_thres 0.02\
                    --det_thres 0.6\
                    --new_overlap_thres 0.85\
                    --sim_thres 0.8\
                    --input_video /usr/users/agecker/datasets/lemur_ids/alpha_ID_validation_snippets/"$vid".mp4\
                    --output_root ../videos/lemur_ids_head/model_"$i"\
                    --output_name "$vid"\
                    --store_opt\
                    --line_thickness 2\
                    --arch hrnet_32\
                    --output_format text\
                    --reid_cls_names lemur,box\
                    --gc_cls_names Cha,Flo,Gen,Geo,Her,Rab,Red,Uns\
                    --proportion_iou 0.2\
                    --double_kalman\
                    --gc_dim 128\
                    --use_gc
                    #--gc_with_roi\
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