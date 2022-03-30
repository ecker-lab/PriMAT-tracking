import os

os.chdir('/home/matthias/monkey/MC-FairMOT_mt/src')

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0" 

os.system("python demo.py mot --load_model ../exp/mot/lab_all/model_90.pth --conf_thres 0.02 --det_thres 0.6 --emb_sim_thres 0.6 --iou_sim_thres 0.5 --input_h 768 --input_w 1024 --output_format 'video' --input_video /media/hdd2/matthias/monkey_vids/hum/trim.mp4 --output_root /media/hdd2/matthias/monkey_vids/hum/out_all_29-3_det06")

