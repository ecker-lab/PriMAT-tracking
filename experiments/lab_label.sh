export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=3

cd src
python demo.py mot --load_model ../exp/mot/lab/model_80.pth --conf_thres 0.02 --det_thres 0.5 --input_video ../../trim.mp4 --output_root ../exp/mot/out2/
cd ..

