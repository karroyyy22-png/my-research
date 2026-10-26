"""用冻结配置重出某个 split 的全部报告（不拟合）。test 仍受 eval_test.py 的哈希保护，这里只允许 val_dev / val_holdout。"""
import sys, json
from common import *
from pipeline import *

sp = sys.argv[1]
assert sp in ('val_dev', 'val_holdout')
cfg = json.load(open(f'{RES_DIR}/frozen_config.json'))['decoders']
split = json.load(open(f'{RES_DIR}/split_val.json'))
val = load_frame('val').set_index('meta_index', drop=False)
report_split(cfg, val.loc[split[sp]].reset_index(drop=True), sp, do_e02=(sp == 'val_holdout'))
