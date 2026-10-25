"""第0步自检：行数、metadata 一致性、四维标签与列顺序、复现已知数字。任何一项不过即非零退出。"""
import sys, json
import numpy as np
from common import *

fails = []
def check(name, ok, info=''):
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {info}")
    if not ok:
        fails.append(name)

out = {}
dfs = {}
for split, n_exp in [('val', 22126), ('test', 50705)]:
    df = load_frame(split)
    dfs[split] = df
    check(f'{split} 行数={n_exp}', len(df) == n_exp, f'(实际 {len(df)})')
    check(f'{split} meta_index 唯一', df.meta_index.is_unique)
    check(f'{split} image 与 metadata 一致', (df.image == df.meta_image).all(), f'(不一致 {(df.image != df.meta_image).sum()})')
    check(f'{split} fake_cls 与 metadata 一致', (df.fake_cls == df.meta_fake_cls).all())
    T = df[[f't_{d}' for d in DIMS]].values
    T_str = np.stack([cls_to_mc(c) for c in df.fake_cls])
    check(f'{split} fake_cls 生成四维标签 == target_mc', np.array_equal(T, T_str), f'(不一致行 {(T != T_str).any(1).sum()})')
    # 列顺序：每一维概率与其同名标签的 AUC 应最高（对角占优）
    from sklearn.metrics import roc_auc_score
    _, P = probs(df)
    auc = np.array([[roc_auc_score(T_str[:, j], P[:, k]) for j in range(4)] for k in range(4)])
    check(f'{split} prob_mc 第k列与第k个标签 AUC 最大(FS,FA,TS,TA)', (auc.argmax(1) == np.arange(4)).all(),
          '\n' + np.array2string(auc, precision=3))
    out[f'{split}_auc_matrix_rows_probcol_cols_label'] = auc.round(4).tolist()
    out[f'{split}_source_counts'] = df.source.value_counts().to_dict()
    out[f'{split}_face_bin_counts'] = df.face_bin.value_counts().to_dict()

df = dfs['test']
pf, P = probs(df)
yb = (df.fake_cls != 'orig').values
bm = binary_metrics(yb, pf >= 0.5)
check('test 二分类 ACC≈0.8558', abs(bm['bin_acc'] - 0.8558) <= 0.001, f"(实际 {bm['bin_acc']:.6f})")
check('test Original 误报=4071', bm['orig_FP'] == 4071, f"(实际 {bm['orig_FP']})")
hm = multilabel_head_metrics(P, df[[f't_{d}' for d in DIMS]].values)
for d, ref in zip(DIMS, [78.97, 81.92, 76.32, 75.35]):
    v = hm[f'head_{d}_F1'] * 100
    check(f'test 头 {d} F1≈{ref}', abs(v - ref) <= 0.1, f'(实际 {v:.4f})')
out.update(test_binary=bm, test_heads=hm, fails=fails)
print('test pred_cls==(prob_fake>=0.5) 一致率:', float(((pf >= 0.5).astype(int) == df.pred_cls.values).mean()))
json.dump(out, open(f'{RES_DIR}/step0_selfcheck.json', 'w'), indent=2, ensure_ascii=False, default=float)
if fails:
    print('自检失败:', fails); sys.exit(1)
print('全部通过')
