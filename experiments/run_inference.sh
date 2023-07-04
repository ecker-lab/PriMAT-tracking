
cd src

for vid in  VID_20210223_123630_0 VID_20210224_115455_0 VID_20210301_105722_0 VID_20210227_133440_0 VID_20210228_154053_0 VID_20210302_103130_0 VID_20210301_151229_0 VID_20210228_160721_0 VID_20210224_114038_0 VID_20210224_115729_0 VID_20210223_123817_0 VID_20210301_145312_0 VID_20210228_153846_0  VID_20210223_123854_0 VID_20210228_153942_0 VID_20210301_143635_0 VID_20210302_103307_0 VID_20210227_133251_0

do

python demo.py mot  --load_model ../exp/mot/macaquecpw_seed2/model_250.pth\
                    --conf_thres 0.02\
                    --det_thres 0.5\
                    --new_overlap_thres 0.7\
                    --sim_thres 0.7\
                    --input_video /usr/users/agecker/datasets/macaque_videos_eval/Videos/"$vid".mp4\
                    --output_root ../videos/video_dump/\
                    --output_name "$vid"\
                    --store_opt\
                    --line_thickness 2\
                    --debug_info\
                    --arch hrnet_32\
                    --output_format 'video'\
                    --reid_cls_names 'macaque'\
                    --proportion_iou 0.8\
                    --buffered_iou 0.2
                    #--seed 42
                    #--min-box-area 100
                    #--use_gc\
                    #--gc_cls_names 'Richard,Kiwi,Timo,Alex,Flo'\
                    # --cat_spec_wh\
                    # --input_h 768\
                    # --input_w 1024\
                    # --id_inline\
done
cd ..

