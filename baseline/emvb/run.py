import os

os.system('cd build && make -j')
os.system('./build/perf_emvb -k 10 -nprobe 4 -thresh 0.4 -out-second-stage 50 -thresh-query 0.5 \
  -n-doc-to-score 100 -queries-id-file /home/bianzheng/emvb-fork/index/qid.txt  \
  -alldoclens-path /home/bianzheng/emvb-fork/index/lotte-500-gnd/doclens.npy \
  -index-dir-path /home/bianzheng/emvb-fork/index/lotte-500-gnd -out-file results_10_lotte-500-gnd.tsv')
