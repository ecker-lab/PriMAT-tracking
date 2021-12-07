#! /bin/sh

cd /usr/users/vogg/FairMOT/


for vid in VID_20210223_123630_0 VID_20210224_114038_0 VID_20210224_115729_0 VID_20210223_123817_0 VID_20210224_115455_0 VID_20210223_123854_0 VID_20210228_153846_0

do
    for c in 0.02
    do
        for e in 0.3
        do
            for i in 0.7 #$(seq 0.8 0.1 0.9)
            do
                for t in 0.5  #proportion embedding
                do
                python src/demo.py mot --load_model models/mcqcpz_1e4/model_120.pth --input_video /usr/users/agecker/datasets/macaque_videos/Videos/"$vid".mp4 --output_root videos/thresholds/"$vid"/mcqcpz120_"${c}"_"${e}"_"${i}"_"${t}" --conf_thres $c --emb_sim_thres $e --det_thres $i --proportion_emb $t --output_format text
                done
            done
        done
    done
done