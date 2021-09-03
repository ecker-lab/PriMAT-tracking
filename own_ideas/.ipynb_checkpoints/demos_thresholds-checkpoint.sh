#! /bin/sh

cd /usr/users/vogg/FairMOT/

for c in $(seq 0.02 0.02 0.08)
do
    for e in $(seq 0.2 0.1 0.7)
    do
        for i in $(seq 0.2 0.1 0.7)
        do
            python src/demo.py mot --load_model models/mcqpose/mcq180.pth --input_video ../test/VID_20210223_123551.mp4 --output_root videos/thres/mcq180_"${c}"_"${e}"_"${i}" --conf_thres $c --emb_sim_thres $e --iou_sim_thres $i
        done
    done
done