export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=3

cd src
python demo.py mot --load_model ../exp/mot/lemur_box/model_last.pth --conf_thres 0.02 --det_thres 0.5 --input_video /home/matthias/storage/sfb1528_storage/Z02/LemursAndBoxes/lemur_videos/Train25_close_1.mp4 --output_root /home/matthias/storage/sfb1528_storage/Z02/LemursAndBoxes/out/Train25_close_1 --store_opt
cd ..

