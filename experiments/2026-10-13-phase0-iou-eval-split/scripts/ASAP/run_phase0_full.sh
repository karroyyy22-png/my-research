#!/bin/bash
source /nas/data_2/wanhongz/miniconda3/etc/profile.d/conda.sh && conda activate ASAP
cd /nas/data_2/wanhongz/ASAP
EXPID=log20260602_134300_train_4gpu_batch16
BK=/nas/data_2/wanhongz/backup_phase0
echo "commit: $(git rev-parse --short HEAD)"
for MODE in none cscl zero; do
  EVAL_DIR=eval_phase0_full_$MODE
  if [ -e results/$EXPID/$EVAL_DIR ]; then
    echo "[STOP] $EVAL_DIR 已存在，跳过"; continue
  fi
  touch $BK/marker_full_$MODE; sleep 1
  echo "[$(date '+%F %T')] start $MODE"
  CUDA_VISIBLE_DEVICES=2 python test.py \
    --config configs/test.yaml --output_dir results --launcher pytorch --rank 0 \
    --log_num $EXPID --dist-url tcp://127.0.0.1:10031 --token_momentum \
    --world_size 1 --test_epoch best \
    --box_postproc $MODE --eval_dirname $EVAL_DIR \
    > $BK/full_$MODE.log 2>&1
  rc=$?
  echo "[$(date '+%F %T')] end $MODE exit=$rc"
  echo "  期间被修改的文件:"
  find results/$EXPID -newer $BK/marker_full_$MODE -type f
done
echo "[$(date '+%F %T')] ALL DONE"
