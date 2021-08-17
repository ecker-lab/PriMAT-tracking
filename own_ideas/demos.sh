#! /bin/sh

cd /usr/users/vogg/FairMOT/

for i in 40 50 60 70 80 90 100 110 120
do
    python src/demo.py mot --load_model models/mix/mcq"${i}"_oms2.pth --input-video ../test/VID_20210223_123551.mp4 --output-root videos/mcq"${i}"oms2 --conf_thres 0.4
done