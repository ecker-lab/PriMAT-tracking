export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0

cd src
python demo.py mot --load_model ../exp/mot/lab_all/model_30.pth --conf_thres 0.2 --det_thres 0.7 --input_video /media/hdd2/matthias/monkey_vids/hum/trim.mp4 --output_root /media/hdd2/matthias/monkey_vids/hum/out_all_new
cd ..

