#! /bin/sh

cd /usr/users/vogg/FairMOT/

count=0
for vid in VID_20210223_123630_0 VID_20210224_114038_0 VID_20210224_115729_0 VID_20210223_123817_0 VID_20210224_115455_0 VID_20210223_123854_0

do
    for i in 3 6 9 12
    do
        python src/demo.py mot --load_model models/"wild$count"/model_"$i".pth --input_video /usr/users/agecker/datasets/macaque_videos/Videos/"$vid".mp4 --output_root videos/new_tracking/"$vid"/"wild$count"_"$i" --output_format text
    done
    count=$((count+1))
done
