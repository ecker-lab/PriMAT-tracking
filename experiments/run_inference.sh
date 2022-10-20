export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0

cd src
python demo.py mot --load_model ../exp/mot/test_pose_high_batch-loss_fix2/model_last.pth --conf_thres 0.02 --det_thres 0.5 --input_video /home/matthias/monkey/trim15.mp4 --output_root /home/matthias/monkey/test_pose_high_batch-loss_fix2/ --store_opt --line_thickness 2 --use_pose #--reid_cls_names 'lemur,box' --reid_cls_ids '0,1' --cat_spec_wh #--use_pose
cd ..

