#! /bin/sh

cd /usr/users/agecker/datasets/macaque_videos_vogg
mkdir predictions

for vid in *.mp4;
do
    cd /usr/users/vogg/FairMOT/

    echo "$vid"
    python src/demo.py mot --load_model models/mcqcpz_1e4/model_120.pth --input_video /usr/users/agecker/datasets/macaque_videos_vogg/"$vid" --output_root /usr/users/agecker/datasets/macaque_videos_vogg/predictions/"$vid" --output_format text --emb_sim_thres 0.3 --conf_thres 0.02
    
    cd /usr/users/agecker/datasets/macaque_videos_vogg
done

