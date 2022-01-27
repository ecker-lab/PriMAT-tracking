#! /bin/sh

cd /usr/users/vogg/FairMOT/

for vid in FSd16D16092012K2V1v1

do
        python src/demo.py mot --load_model models/mcqcpz_1e4/model_120.pth --input_video /usr/users/agecker/datasets/Videos_redfronted_lemurs/2012/"$vid".MPG --output_root videos/lemurs/"$vid" --output_format video --emb_sim_thres 0.3
done

