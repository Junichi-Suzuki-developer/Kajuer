#!/bin/bash

mkdir -p logs

export MKL_DYNAMIC=FALSE
export MKL_NUM_THREADS=1

(rm -rf spike_records) && \
(mkdir spike_records) && \
(rm -rf weight_map_dir) && \
(mkdir weight_map_dir) && \
(rm -f fire_record.txt) && \
(rm -f result.txt) && \
(icpx -xhost -qmkl=sequential -ipo -O3 -g main/main.cpp -I./Kajuer -std=c++23) && \
(./a.out > out.txt 2> err.txt) && \
(grep -a answer out.txt >> result.txt) && \
(grep -a label_answer out.txt) | \
(awk 'BEGIN{answer["true"]=0; answer["false"]=0;} {answer[$4] += 1} END{ print answer["true"], answer["false"]}' >> result.txt)
export dirname=`date '+%Y%m%d%H%M%S'`
if [ -f ./result.txt ]
then
    mkdir -p logs/${dirname}
    cp -r spike_records weight_map_dir Kajuer/Param/Param.h result.txt fire_record.txt logs/${dirname}
fi
