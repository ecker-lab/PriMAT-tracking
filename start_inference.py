import os

os.chdir('/home/matthias/monkey/MC-FairMOT_mt/src')

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3" 
from pathlib import Path
p = Path('/media/hdd2/matthias/sfb1528_storage/Z01/PlaygroundSessions4ActionClassification/')
q = Path('/media/hdd2/matthias/sfb1528_storage/Z01/Output_Pose/')
import shutil

for inp in p.glob('luk*69.mp4'):

    # get rid of .mp4
    # dir_in = p / inp.stem
    # if dir_in.is_dir():
    #     continue
    # dir_in.mkdir()
    # dir_in = str(dir_in)

    # os.system("ffmpeg -i "+str(inp)+" -c copy -map 0 -segment_time 00:05:00 -f segment -reset_timestamps 1 "+dir_in+"/vid%03d.mp4")

    # add video name without .mp4 to output path
    dir_out = q / inp.stem
    if not dir_out.is_dir():
        dir_out.mkdir()
    dir_out = str(dir_out)
    
    # for inp2 in (p / inp.stem).glob('vid*.mp4'):
        # ou = Path(dir_out, inp2.name)
        # os.system("python demo.py mot --load_model ../models/mcFairmotPose.pth --conf_thres 0.2 --det_thres 0.7 --emb_sim_thres 0.1 --iou_sim_thres 0.1 --input_video "+str(inp2)+" --output_root "+str(ou))
    if len(os.listdir(dir_out)) == 0: # Check is empty..
        os.system(f"python demo.py mot-p --load_model ../models/mcFairmotPose_without-val.pth --conf_thres 0.2 --det_thres 0.7 --emb_sim_thres 0.1 --iou_sim_thres 0.1 --input_video {str(inp)} --output_root {str(dir_out)} --output_format text --use_pose")
    else:
        print('not this one==============================')