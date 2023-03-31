#! /bin/sh

cd /usr/users/vogg/monkey-tracking-in-the-wild/src


for vid in VID_20210301_105722_0 VID_20210223_123630_0 VID_20210227_133440_0 VID_20210228_154053_0 VID_20210302_103130_0 VID_20210301_151229_0 VID_20210228_160721_0 VID_20210224_114038_0 VID_20210224_115729_0 VID_20210223_123817_0 VID_20210301_145312_0 VID_20210228_153846_0 VID_20210224_115455_0 VID_20210223_123854_0 VID_20210228_153942_0 VID_20210301_143635_0 VID_20210302_103307_0 VID_20210227_133251_0

do
    for i in 23 100
    do
    
    
python demo.py mot  --load_model ../exp/mot/al_macaquecpw/model_"$i".pth\
                    --conf_thres 0.01\
                    --det_thres 0.6\
                    --input_video /usr/users/agecker/datasets/macaque_videos_eval/Videos/"$vid".mp4\
                    --output_root ../videos/sharmita/"$vid"/output_"$i"\
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
cd ..

