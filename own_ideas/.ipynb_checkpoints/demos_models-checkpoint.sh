#! /bin/sh

cd /usr/users/vogg/FairMOT/

for i in 210 220
do
    python src/demo.py mot --load_model models/mcqcp/mcqcp"${i}".pth --input_video ../test/VID_20210223_123551.mp4 --output_root videos/mcqcp"${i}" --conf_thres 0.4
done