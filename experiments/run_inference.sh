export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0

cd src
python demo.py mot --load_model ../exp/mot/lab_all_pose_2ndTry/model_last.pth --conf_thres 0.02 --det_thres 0.5 --input_video /home/matthias/monkey/trim15.mp4 --output_root /home/matthias/monkey/richard_test-pose --store_opt --use_pose
cd ..

