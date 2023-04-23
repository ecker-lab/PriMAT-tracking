#! /bin/sh

cd /usr/users/vogg/monkey-tracking-in-the-wild/src


for seed in 1 2 3
do

for vid in VID_20210301_105722_0 VID_20210223_123630_0 VID_20210227_133440_0 VID_20210228_154053_0 VID_20210302_103130_0 VID_20210301_151229_0 VID_20210228_160721_0 VID_20210224_114038_0 VID_20210224_115729_0 VID_20210223_123817_0 VID_20210301_145312_0 VID_20210228_153846_0 VID_20210224_115455_0 VID_20210223_123854_0 VID_20210228_153942_0 VID_20210301_143635_0 VID_20210302_103307_0 VID_20210227_133251_0

do
    for i in 10 20 30 40 50 60 70 80 90 100 110 120 130 140 150 160 170 180 190 200 210 220 230 240 250
    do
    
    
    for dataset in macaquepose
    do
    out=$((i * 2))

python demo.py mot  --load_model ../exp/mot/"$dataset"_seed"$seed"/model_"$out".pth\
                    --conf_thres 0.02\
                    --det_thres 0.6\
                    --proportion_iou 0.8\
                    --new_overlap_thres 0.8\
                    --sim_thres 0.8\
                    --input_video /usr/users/vogg/macaque_videos_eval/Videos/"$vid".mp4\
                    --output_root ../videos/methods_paper/"$vid"/"$dataset"_"$seed"_"$i"\
                    --output_name 'testing'\
                    --store_opt\
                    --line_thickness 2\
                    --debug_info\
                    --arch hrnet_32\
                    --output_format text\
                    --reid_cls_names 'macaque'\
                    #--seed 42
                    #--min-box-area 100
                    #--use_gc\
                    #--gc_cls_names 'Richard,Kiwi,Timo,Alex,Flo'\
                    # --cat_spec_wh\
                    # --input_h 768\
                    # --input_w 1024\
                    # --id_inline\
    done
    
    for dataset in macaquecpw macaquecp
    do
    out=$i

python demo.py mot  --load_model ../exp/mot/"$dataset"_seed"$seed"/model_"$out".pth\
                    --conf_thres 0.02\
                    --det_thres 0.6\
                    --proportion_iou 0.8\
                    --new_overlap_thres 0.8\
                    --sim_thres 0.8\
                    --input_video /usr/users/vogg/macaque_videos_eval/Videos/"$vid".mp4\
                    --output_root ../videos/methods_paper/"$vid"/"$dataset"_"$seed"_"$i"\
                    --output_name 'testing'\
                    --store_opt\
                    --line_thickness 2\
                    --debug_info\
                    --arch hrnet_32\
                    --output_format text\
                    --reid_cls_names 'macaque'\
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
cd ..

