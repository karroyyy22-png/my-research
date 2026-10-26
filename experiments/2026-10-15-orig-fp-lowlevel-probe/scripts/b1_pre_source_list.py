"""B1 读图前：用该 split 全量协变量（来源/脸桶/W/H 与 FP 标签，不读图）算各来源主规则匹配率，确定 v3 次要分析来源名单并落盘。
与 b1_lowlevel.py 第 206–215 行同一算法（直接复用其函数与 seed），读图后应与 matching.csv 的 full_split_covariates_only 行一致。
用法: python b1_pre_source_list.py --split val|test
输出 results/B1/<split>/pre_read_source_list.csv
"""
import os, sys, argparse
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from b_common import check_frozen, load_split, out_dir  # noqa: E402
from b1_lowlevel import face_geom, bucket, match_pairs  # noqa: E402


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', choices=['val', 'test'], default='val')
    a = ap.parse_args(argv)
    cfg, sha = check_frozen(a.split)
    seed = cfg['seed']
    od = out_dir('B1', a.split, False)

    df, _ = load_split(cfg, a.split)
    geo = [face_geom(m) for m in df.mtcnn]
    df['max_face'] = [g[0] for g in geo]
    df['bucket'] = [bucket(mf, n) for mf, n in geo]
    df['key'] = list(zip(df.source, df.bucket, df.W, df.H))
    o_all = df[df.fake_cls == 'orig'].copy(); o_all['fp'] = (o_all.prob_fake >= 0.5).astype(int)
    full_pairs = match_pairs(o_all[o_all.fp == 1], o_all[o_all.fp == 0], seed)
    fm = {i for i, _ in full_pairs}
    rows = [dict(scope='full_split_covariates_only', n_fp=int(o_all.fp.sum()), n_matched=len(full_pairs),
                 match_rate=len(full_pairs) / max(1, o_all.fp.sum()))]
    rows += [dict(scope=f'full_split_covariates_only:{s}', n_fp=len(g), n_matched=int(g.index.isin(fm).sum()),
                  match_rate=g.index.isin(fm).mean()) for s, g in o_all[o_all.fp == 1].groupby('source')]
    r = pd.DataFrame(rows)
    r['secondary_source'] = [(':' in s) and m < 0.70 for s, m in zip(r.scope, r.match_rate)]
    r['split'] = a.split; r['frozen_sha256'] = sha
    r.to_csv(f'{od}/pre_read_source_list.csv', index=False)
    print(r.to_string(index=False))
    print('[low_sources]', [s.split(':', 1)[1] for s in r.scope[r.secondary_source]])
    print(f'[done] {od}/pre_read_source_list.csv')


if __name__ == '__main__':
    main(sys.argv[1:])
