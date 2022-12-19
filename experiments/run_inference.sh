export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=1

cd src
python demo.py mot --load_model ../exp/mot/with_WH-lemur_box-7-11-22/model_last.pth --conf_thres 0.02 --det_thres 0.5 --input_video /home/matthias/monkey/lemur_box-trim15.mp4 --output_root /home/matthias/monkey/with_cat2/ --store_opt --line_thickness 2 --reid_cls_names 'lemur,box' --reid_cls_ids '0,1' --cat_spec_wh #--use_pose
cd ..

