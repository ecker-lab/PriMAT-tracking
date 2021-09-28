#! /bin/sh

cd /usr/users/vogg/FairMOT/

for vid in VID_20210223_123630_0 VID_20210223_123817_0 VID_20210223_123854_0
do
    for i in 100 120 140 160 180 200 220 240 260 280
    do
        python src/demo.py mot --load_model models/mcqcpz/mcqcpz"$i".pth --input_video /usr/users/agecker/datasets/macaque_videos/Videos/"$vid".mp4 --output_root videos/new_tracking/"$vid"/mcqcpz"$i"
    done
done
