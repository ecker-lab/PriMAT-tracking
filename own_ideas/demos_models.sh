#! /bin/sh

cd /usr/users/vogg/FairMOT/

for vid in VID_20210223_123630_0 #VID_20210224_114038_0 VID_20210224_115729_0 VID_20210223_123817_0 VID_20210224_115455_0 VID_20210223_123854_0

do
    for i in 150 #10 20 30 40 50 60 70 80 90 100 110 120 130 140 150 160 170 180 190 200 210 220 230 240 250
    do
        python src/demo.py mot --load_model models/mcqcp/model_"$i".pth --input_video /usr/users/agecker/datasets/macaque_videos/Videos/"$vid".mp4 --output_root videos/new_tracking/"$vid"/mcqcpkalman"$i" --output_format text
    done
done
