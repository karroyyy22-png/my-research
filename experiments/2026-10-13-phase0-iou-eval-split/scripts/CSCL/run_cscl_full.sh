#!/bin/bash
cd /nas/data_2/wanhongz/CSCL/code/MultiModal-DeepFake-main
PY=/nas/data_2/wanhongz/miniconda3/envs/CSCL/bin/python
BK=/nas/data_2/wanhongz/backup_phase0
unset PYTHONPATH
echo "host: $(hostname)  commit: $(git rev-parse --short HEAD)"
touch $BK/cscl/marker_full_start
PORT=23459
for MODE in cscl none zero; do
  EVAL_DIR=eval_phase0_full_$MODE
  if [ -e results/CSCL/$EVAL_DIR ]; then
    echo "[STOP] $EVAL_DIR 已存在，跳过"; PORT=$((PORT+1)); continue
  fi
  (
    echo "[$(date '+%F %T')] start $MODE (port $PORT)"
    CUDA_VISIBLE_DEVICES=0 $PY test.py \
      --config configs/test_phase0.yaml --output_dir ./results --launcher pytorch --rank 0 \
      --log_num CSCL --dist-url tcp://localhost:$PORT --world_size 1 --test_epoch 49 \
      --box_postproc $MODE --eval_dirname $EVAL_DIR \
      > $BK/cscl/full_$MODE.log 2>&1
    rc=$?
    echo "[$(date '+%F %T')] end $MODE exit=$rc"
  ) &
  PORT=$((PORT+1))
  sleep 30
done
wait
echo "[$(date '+%F %T')] ALL DONE"
echo "期间被修改的文件:"
find results/CSCL -newer $BK/cscl/marker_full_start -type f
