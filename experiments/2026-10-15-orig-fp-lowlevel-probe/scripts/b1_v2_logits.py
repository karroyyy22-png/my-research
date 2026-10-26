"""V2：同 id 兄弟（原图 vs FS/FA）的 own-dim logit 关系（只读 pkl 与 metadata，不读图；纯描述，见 frozen_hypotheses.json 的 V2 段）。
不能据此断言模型内部机制。
用法: python b1_v2_logits.py --split val|test
输出 results/B1_V2/<split>/：pairs.csv  a_corr.csv  b_delta_logit.csv  c_fp_subset.csv
"""
import os, sys, argparse
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from b_common import check_frozen, load_split, out_dir  # noqa: E402
from b1_v1_siblings import method_of  # noqa: E402

NOTE = '纯描述；不能据此断言模型内部机制'
CLIP_LOGIT = float(np.log((1 - 1e-6) / 1e-6))  # ≈13.8155
GROUP_SRC = 'FP 分组口径：2026-10-07-asap-failure-analysis/scripts/diagnose_extra.py，阈值 0.5；字母同 verify_pairs.py'


def logit(p):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def id_boot(d, stat, B, rng):
    """以 id 为簇重采样 DataFrame 的行，返回 stat(子表) 的 95% CI。"""
    ids, inv = np.unique(d.id.values, return_inverse=True)
    bs = []
    for _ in range(B):
        w = np.bincount(rng.integers(0, len(ids), len(ids)), minlength=len(ids))[inv]
        bs.append(stat(d.iloc[np.repeat(np.arange(len(d)), w)]))
    return float(np.nanpercentile(bs, 2.5)), float(np.nanpercentile(bs, 97.5))


def dist_row(d, col, B, rng, **kw):
    lo, hi = id_boot(d, lambda x: x[col].median(), B, rng)
    return dict(**kw, n_pairs=len(d), n_ids=d.id.nunique(), median=d[col].median(), q1=d[col].quantile(.25),
                q3=d[col].quantile(.75), median_ci_lo=lo, median_ci_hi=hi, note=NOTE)


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', choices=['val', 'test'], default='val')
    a = ap.parse_args(argv)
    cfg, sha = check_frozen(a.split)
    B = cfg['bootstrap']['B']
    rng = np.random.default_rng(cfg['seed'])
    od = out_dir('B1_V2', a.split, False)

    df, _ = load_split(cfg, a.split)
    o = df[df.fake_cls == 'orig'][['id', 'meta_index', 'prob_fake', 'pmc_FS', 'pmc_FA', 'pmc_TS', 'pmc_TA']].copy()
    # FP 分组（口径同 2026-10-07 diagnose_extra.py；字母同 verify_pairs.py），阈值 0.5
    img_t = (o.pmc_FS >= .5) | (o.pmc_FA >= .5); txt_t = (o.pmc_TS >= .5) | (o.pmc_TA >= .5)
    o['fp_group'] = np.select([img_t & ~txt_t, ~img_t & txt_t, img_t & txt_t], ['A', 'B', 'C'], default='D')
    s = df[df.fake_cls.isin(['face_swap', 'face_attribute'])].copy()
    s['method'] = [method_of(p, c) for p, c in zip(s.image, s.fake_cls)]
    s['sib_cls'] = s.fake_cls.map({'face_swap': 'FS', 'face_attribute': 'FA'})
    p = o.add_prefix('o_').rename(columns={'o_id': 'id'}).merge(
        s[['id', 'meta_index', 'prob_fake', 'pmc_FS', 'pmc_FA', 'method', 'sib_cls']].add_prefix('s_').rename(
            columns={'s_id': 'id'}), on='id')
    own = np.where(p.s_sib_cls == 'FS', 'FS', 'FA')
    p['o_p_own'] = np.where(own == 'FS', p.o_pmc_FS, p.o_pmc_FA)
    p['s_p_own'] = np.where(own == 'FS', p.s_pmc_FS, p.s_pmc_FA)
    p['o_logit'], p['s_logit'] = logit(p.o_p_own), logit(p.s_p_own)
    p['d_logit'] = p.s_logit - p.o_logit
    p['o_fp'] = (p.o_prob_fake >= 0.5).astype(int)
    p['clipped'] = (p.o_logit.abs() >= CLIP_LOGIT - 1e-9) | (p.s_logit.abs() >= CLIP_LOGIT - 1e-9)
    p.to_csv(f'{od}/pairs.csv', index=False)

    # (a) 相关；(b) Δlogit 分布。analysis=main 为全部配对，sensitivity 为去掉截断值配对后（v7 补充）
    rows_a, rows_b = [], []
    rho = lambda x: spearmanr(x.o_logit, x.s_logit).correlation
    r2 = lambda x: np.corrcoef(x.o_logit, x.s_logit)[0, 1] ** 2
    for analysis, pp in (('main', p), ('sensitivity_drop_clipped（补充）', p[~p.clipped])):
        for name, d in [('all', pp)] + [(f'sib_cls={k}', g) for k, g in pp.groupby('s_sib_cls')]:
            n_tot = len(p) if name == 'all' else int((p.s_sib_cls == name.split('=')[1]).sum())
            lo1, hi1 = id_boot(d, rho, B, rng); lo2, hi2 = id_boot(d, r2, B, rng)
            rows_a.append(dict(analysis=analysis, stratum=name, n_pairs=len(d), n_ids=d.id.nunique(),
                               frac_dropped=1 - len(d) / n_tot, spearman=rho(d), spearman_ci_lo=lo1,
                               spearman_ci_hi=hi1, pearson_r2=r2(d), r2_ci_lo=lo2, r2_ci_hi=hi2, note=NOTE))
        for name, d in [('all', pp)] + [(f'method={k}', g) for k, g in pp.groupby('s_method')]:
            n_tot = len(p) if name == 'all' else int((p.s_method == name.split('=')[1]).sum())
            rows_b.append(dist_row(d, 'd_logit', B, rng, analysis=analysis, stratum=name, frac_dropped=1 - len(d) / n_tot))
    pd.DataFrame(rows_a).to_csv(f'{od}/a_corr.csv', index=False)
    pd.DataFrame(rows_b).to_csv(f'{od}/b_delta_logit.csv', index=False)

    # (c) 原图为 FP 的子集：全部、按 FP 分组 A/B/C/D（v7）、按方法
    f = p[p.o_fp == 1]
    rows = []
    for name, d in [('all', f)] + [(f'fp_group={k}', g) for k, g in f.groupby('o_fp_group')] + \
            [(f'method={k}', g) for k, g in f.groupby('s_method')]:
        if len(d) == 0:
            continue
        hi_own = lambda x: (x.s_p_own >= 0.5).mean()
        hi_bin = lambda x: (x.s_prob_fake >= 0.5).mean()
        l1, h1 = id_boot(d, hi_own, B, rng); l2, h2 = id_boot(d, hi_bin, B, rng)
        r = dist_row(d, 'd_logit', B, rng, stratum=name)
        r.update(frac_sib_own_ge05=hi_own(d), own_ci_lo=l1, own_ci_hi=h1, frac_sib_own_lt05=1 - hi_own(d),
                 frac_sib_probfake_ge05_ref=hi_bin(d), probfake_ci_lo=l2, probfake_ci_hi=h2, group_source=GROUP_SRC)
        rows.append(r)
    pd.DataFrame(rows).to_csv(f'{od}/c_fp_subset.csv', index=False)

    pd.set_option('display.width', 250); pd.set_option('display.max_columns', 30)
    for fn in ['a_corr', 'b_delta_logit', 'c_fp_subset']:
        print(pd.read_csv(f'{od}/{fn}.csv').drop(columns='note').round(4).to_string(index=False))
    print(f'[done] {od}  frozen_sha256={sha}  ({NOTE})')


if __name__ == '__main__':
    main(sys.argv[1:])
