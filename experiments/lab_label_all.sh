export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=3

cd src
for folder in /media/hdd2/matthais/monkey_vids/
do
	for vid in $folder/*
	do
		filename=$(basename "$vid")
		python demo.py mot --load_model ../exp/mot/lab/model_80.pth --conf_thres 0.02 --det_thres 0.5 --input_video echo $vid --output_root echo $folder "/out"
	done
done
cd ..

