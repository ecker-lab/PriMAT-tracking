export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=2

cd src
python demo.py mot --load_model ../exp/mot/lab_all_pose_2ndTry/model_30.pth --conf_thres 0.2 --det_thres 0.7 --input_video /home/matthias/monkey/trim15.mp4 --output_root /home/matthias/monkey/trial_inference-softmax
cd ..

