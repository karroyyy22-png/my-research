"""B1 H1 判定：机械实现 frozen_hypotheses.json 的 B1.criteria_v3（not_supported 与 holm_version 沿用 B1.criteria）。
该脚本在看过 test 之后编写，仅机械实现事先冻结的判据；不读图，只读 b1_lowlevel.py 输出的 split_status.csv。
- 有方向的 7 个特征（predictions_v3.directed）：support_H1 / opposite_to_H1 / not_supported / insufficient，
  未校正与 Holm 校正两版并列。只给 --val 时 test 条件记缺失，因此不可能判 support_H1。
- 无方向的 4 个特征（criteria_v3.undirected_features）：只用 v3 的两种标签。val 与 test 都满足
  “AUC 95% CI 不含 0.5 且 |AUC−0.5|>=0.10” 才标“双侧有差异”，否则标“无可分辨差异”（只给 --val 时一律后者）；
  Holm 版按 criteria.holm_version 把“CI 不含 0.5”换成 Holm p<0.05，两版并列。
  另列 ci_in_045_055（各 split 的 AUC CI 是否全在 [0.45,0.55] 内），仅作参考，不参与判定。
- jpeg_q 为常数时按 predictions_v3.jpeg_q 标“不可检验”。
用法: python b1_verdict.py --val results/B1/val_v5 [--test results/B1/test] --out results/B1_verdict/<name>
"""
import os, sys, argparse
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from b_common import check_frozen  # noqa: E402

COLS = ['auc', 'auc_ci_lo', 'auc_ci_hi', 'p_mwu', 'p_holm', 'coef', 'coef_ci_lo', 'coef_ci_hi', 'p_coef', 'p_coef_holm']


def directed_verdict(r, pred, has_test, holm):
    dev = r.val_auc - 0.5
    if r.val_auc_ci_lo >= 0.45 and r.val_auc_ci_hi <= 0.55:
        return 'not_supported'
    if np.sign(dev) == -pred and abs(dev) >= 0.10:
        return 'opposite_to_H1'
    auc_sig = (r.val_p_holm < 0.05) if holm else (r.val_auc_ci_lo > 0.5 or r.val_auc_ci_hi < 0.5)
    coef_sig = (r.val_p_coef_holm < 0.05) if holm else (r.val_coef_ci_lo > 0 or r.val_coef_ci_hi < 0)
    test_dir = has_test and np.sign(r.test_auc - 0.5) == pred
    if np.sign(dev) == pred and abs(dev) >= 0.10 and auc_sig and test_dir and coef_sig and np.sign(r.val_coef) == pred:
        return 'support_H1'
    return 'insufficient'


def undirected_verdict(r, has_test, holm):
    def ok(s):
        sig = (r[f'{s}_p_holm'] < 0.05) if holm else (r[f'{s}_auc_ci_lo'] > 0.5 or r[f'{s}_auc_ci_hi'] < 0.5)
        return sig and abs(r[f'{s}_auc'] - 0.5) >= 0.10
    return '双侧有差异' if has_test and ok('val') and ok('test') else '无可分辨差异'


def ci_in_band(r, has_test):
    splits = ['val', 'test'] if has_test else ['val']
    return ';'.join(f"{s}={'是' if r[f'{s}_auc_ci_lo'] >= 0.45 and r[f'{s}_auc_ci_hi'] <= 0.55 else '否'}" for s in splits)


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument('--val', required=True)
    ap.add_argument('--test', default='')
    ap.add_argument('--out', required=True)
    a = ap.parse_args(argv)
    cfg, sha = check_frozen('test' if a.test else 'val')
    p3 = cfg['B1']['predictions_v3']
    pred = {q: (-1 if '更低' in s else 1) for q, s in p3['directed'].items()}
    undirected = [q for q in p3['undirected'] if q in cfg['B1']['multiple_testing']['family']]

    v = pd.read_csv(os.path.join(a.val, 'split_status.csv'))
    t = pd.DataFrame(columns=['feature', *COLS])
    if a.test:
        t = pd.read_csv(os.path.join(a.test, 'split_status.csv'))
    m = v[['feature', *COLS]].add_prefix('val_').rename(columns={'val_feature': 'feature'}).merge(
        t[['feature', *COLS]].add_prefix('test_').rename(columns={'test_feature': 'feature'}), on='feature', how='left')
    m['test_present'] = bool(a.test)

    rows = []
    for _, r in m.iterrows():
        q = r.feature
        if q in pred:
            rows.append(dict(feature=q, kind='directed', pred_dir=pred[q],
                             verdict=directed_verdict(r, pred[q], bool(a.test), False),
                             verdict_holm=directed_verdict(r, pred[q], bool(a.test), True), **r.drop('feature').to_dict()))
        elif q in undirected:
            rows.append(dict(feature=q, kind='undirected', pred_dir=0,
                             verdict=undirected_verdict(r, bool(a.test), False),
                             verdict_holm=undirected_verdict(r, bool(a.test), True),
                             ci_in_045_055=ci_in_band(r, bool(a.test)), **r.drop('feature').to_dict()))
    const = str(v.constant_excluded.iloc[0]) if 'constant_excluded' in v and v.constant_excluded.notna().any() else ''
    for q in [c for c in const.split(',') if c]:
        rows.append(dict(feature=q, kind='constant', verdict='不可检验（常数）', verdict_holm='不可检验（常数）',
                         note=p3['jpeg_q'] if q == 'jpeg_q' else ''))
    out = pd.DataFrame(rows)
    out['val_dir'] = a.val; out['test_dir'] = a.test; out['frozen_sha256'] = sha
    out['label'] = '该脚本在看过 test 之后编写，仅机械实现事先冻结的判据'
    os.makedirs(a.out, exist_ok=True)
    out.to_csv(os.path.join(a.out, 'h1_verdict.csv'), index=False)
    pd.set_option('display.width', 250); pd.set_option('display.max_columns', 30)
    show = ['feature', 'kind', 'pred_dir', 'val_auc', 'val_auc_ci_lo', 'val_auc_ci_hi', 'val_coef', 'test_auc', 'verdict', 'verdict_holm', 'ci_in_045_055']
    print(out[show].round(3).to_string(index=False))
    print(out[out.kind == 'directed'].verdict.value_counts().to_string())
    print(out[out.kind == 'undirected'].verdict.value_counts().to_string())
    print(f'[done] {a.out}/h1_verdict.csv  frozen_sha256={sha}')


if __name__ == '__main__':
    main(sys.argv[1:])
