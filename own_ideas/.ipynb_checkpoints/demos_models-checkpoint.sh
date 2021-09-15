#! /bin/sh

cd /usr/users/vogg/FairMOT/

for vid in VID_20210223_123817_0 VID_20210223_123854_0
do
    for i in 10 20 30 40 50 60 70 80 90 100 110 120 130 140 150 160 170 180 190 200 210 220
    do
        python src/demo.py mot --load_model models/mcqcp/mcqcp"$i".pth --input_video /usr/users/agecker/datasets/macaque_videos/Videos/"$vid".mp4 --output_root videos/"$vid"/mcqcp"$i" --conf_thres 0.4
    done
done