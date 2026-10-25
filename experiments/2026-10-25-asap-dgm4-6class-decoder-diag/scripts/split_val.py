"""val 按 fake_cls 分层、seed=2026 随机对半分为 val_dev / val_holdout，保存 meta_index 清单。已存在则拒绝覆盖。"""
import os, json
import numpy as np
from common import *

SEED = 2026
path = f'{RES_DIR}/split_val.json'
assert not os.path.exists(path), f'{path} 已存在，划分一经写死不再重做'
df = load_frame('val')
rng = np.random.RandomState(SEED)
dev, hold = [], []
for c in CLASSES9:
    idx = np.sort(df.meta_index[df.fake_cls == c].values)
    perm = rng.permutation(idx)
    k = len(idx) // 2
    dev += perm[:k].tolist(); hold += perm[k:].tolist()
split = dict(seed=SEED, method='按 fake_cls 分层，每类内 np.random.RandomState(2026).permutation 后前 floor(n/2) 为 val_dev，其余为 val_holdout；类顺序为 CLASSES9',
             val_dev=sorted(int(i) for i in dev), val_holdout=sorted(int(i) for i in hold))
json.dump(split, open(path, 'w'))
d = df.set_index('meta_index').fake_cls
print('val_dev', len(dev), d.loc[dev].value_counts().to_dict())
print('val_holdout', len(hold), d.loc[hold].value_counts().to_dict())
