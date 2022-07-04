import os

os.chdir('/home/matthias/monkey/MC-FairMOT_mt/src')

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2" 
from pathlib import Path
p = Path('/media/hdd2/matthias/monkey_vids/luk/')
q = Path('/media/hdd2/matthias/monkey_vids_output_pose/luk/')
for inp in p.glob('*.mp4'):
    dir_in = p / inp.stem
    # if dir_in.is_dir():
    #     continue
    # dir_in.mkdir()
    dir_in = str(dir_in)
    # os.system("ffmpeg -i "+str(inp)+" -c copy -map 0 -segment_time 00:05:00 -f segment -reset_timestamps 1 "+dir_in+"/vid%03d.mp4")
    dir_out = q / inp.stem
    if not dir_out.is_dir():
        dir_out.mkdir()
    dir_out = str(dir_out)
    for inp2 in (p / inp.stem).glob('vid*.mp4'):
        ou = Path(dir_out, inp2.name)
        os.system("python demo.py mot --load_model ../exp/mot/lab_all_pose_2ndTry/model_last.pth --conf_thres 0.2 --det_thres 0.7 --emb_sim_thres 0.1 --iou_sim_thres 0.1 --input_video "+str(inp2)+" --output_root "+str(ou))

