export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=1,3

cd src
python demo.py mot --gpus '0,1' --load_model ../exp/mot/lab_all/model_last.pth --conf_thres 0.2 --det_thres 0.7 --input_video /media/hdd2/matthias/monkey_vids/test_trains/trim_short_short.mp4 --output_root /media/hdd2/matthias/monkey_vids/test_trains/out_trim_short_short
cd ..

