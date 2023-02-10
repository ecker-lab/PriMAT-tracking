export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=3

cd src
python demo.py mot  --load_model ../exp/mot/testing_fixes_gc/model_last.pth\
                    --conf_thres 0.02\
                    --det_thres 0.5\
                    --input_video /home/matthias/monkey/trim1.mp4\
                    --output_root /home/matthias/monkey/testing_fixes_gc_text/\
                    --output_name 'testing_fixes_gc_text'\
                    --store_opt\
                    --use_gc\
                    --gc_cls_names 'Richard,Kiwi,Timo,Alex,Flo'\
                    --line_thickness 2\
                    --debug_info\
                    # --id_inline\
                    # --output_format 'text'\
                    # --cat_spec_wh\
                    # --input_h 768\
                    # --input_w 1024\
                    # --reid_cls_names 'lemur,box'\
                    #--seed 42
                    #--min-box-area 100
cd ..

