"""用 results/frozen_config.json 在 test 上评估一次。不拟合任何东西。

重复运行只允许在冻结配置哈希不变时（结果确定性复现）；配置哈希变了就拒绝。
"""
import os, json, hashlib, datetime
from common import *
from pipeline import *

frozen = json.load(open(f'{RES_DIR}/frozen_config.json'))
cfg = frozen['decoders']
sha = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()
assert sha == frozen['sha256_of_decoders'], 'frozen_config 的解码器参数被改动过'
marker = f'{RES_DIR}/test_evaluated.json'
if os.path.exists(marker):
    assert json.load(open(marker))['sha256_of_decoders'] == sha, 'test 已用另一份配置评估过，拒绝再评估'

test = load_frame('test')
res = report_split(cfg, test, 'test')
json.dump(dict(sha256_of_decoders=sha, time=datetime.datetime.now().isoformat(timespec='seconds'), n=len(test)),
          open(marker, 'w'), indent=2)
print(res['e01']['main'][['decoder', 'macroF1_6', 'balAcc_6', 'acc_6', 'orig_R']].round(4).to_string(index=False))
print(res['e01']['binary'].round(4).to_string(index=False))
