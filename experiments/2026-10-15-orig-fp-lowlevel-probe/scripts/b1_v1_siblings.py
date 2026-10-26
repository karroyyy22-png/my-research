"""V1：同 id 兄弟样本的头分数差（只读 pkl 与 metadata，不读图；纯描述，见 frozen_hypotheses.json 的 V1 段）。
不能据此断言模型内部是否使用跨模态特征。
用法: python b1_v1_siblings.py --split val|test [--n 500]
输出 results/B1_V1/<split>/：pairs_caption_sib.csv  pairs_image_sib.csv  a_caption_sib_text_dims.csv
     b_image_sib_image_dims.csv  c_rank_caption_sib.csv  d_coverage.csv
"""
import os, sys, argparse
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from b_common import check_frozen, load_split, out_dir  # noqa: E402

NOTE = '纯描述；不能据此断言模型内部是否使用跨模态特征'


def method_of(path, fake_cls):
    if '/manipulation/' in path:
        return path.split('/')[2]
    return fake_cls


def cluster_boot(d, col, stat, B, rng):
    """以 id 为簇重采样，返回 stat(值) 的 95% CI。"""
    g = {k: v[col].values for k, v in d.groupby('id')}
    keys = np.array(list(g))
    if len(keys) == 0:
        return np.nan, np.nan
    bs = [stat(np.concatenate([g[k] for k in keys[rng.integers(0, len(keys), len(keys))]])) for _ in range(B)]
    return float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def delta_table(pairs, dims, strata, B, rng):
    rows = []
    for sname, sub in [('all', pairs)] + [(f'{strata}={k}', g) for k, g in pairs.groupby(strata)]:
        for dm in dims:
            d = sub[['id', f'd_{dm}']].rename(columns={f'd_{dm}': 'v'}).dropna()
            d['a'] = d.v.abs()
            if len(d) == 0:
                continue
            lo, hi = cluster_boot(d, 'a', np.median, B, rng)
            rows.append(dict(stratum=sname, dim=dm, n_pairs=len(d), n_ids=d.id.nunique(),
                             abs_median=d.a.median(), abs_q1=d.a.quantile(.25), abs_q3=d.a.quantile(.75),
                             abs_median_ci_lo=lo, abs_median_ci_hi=hi, signed_median=d.v.median(),
                             signed_mean=d.v.mean(), note=NOTE))
    return pd.DataFrame(rows)


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', choices=['val', 'test'], default='val')
    ap.add_argument('--n', type=int, default=0, help='小样本：随机 n 条 Original（兄弟池仍为全 split）')
    a = ap.parse_args(argv)
    cfg, sha = check_frozen(a.split)
    seed, B = cfg['seed'], cfg['bootstrap']['B']
    rng = np.random.default_rng(seed)
    od = out_dir('B1_V1', a.split, a.n > 0)

    df, _ = load_split(cfg, a.split)
    df['method'] = [method_of(p, c) for p, c in zip(df.image, df.fake_cls)]
    o = df[df.fake_cls == 'orig']
    if a.n:
        o = o.sample(n=min(a.n, len(o)), random_state=seed)
    o = o.assign(fp=(o.prob_fake >= 0.5).astype(int))
    cols = ['id', 'meta_index', 'image', 'fake_cls', 'method', 'prob_fake', 'pmc_FS', 'pmc_FA', 'pmc_TS', 'pmc_TA']
    osel = o[cols + ['fp']].add_prefix('o_').rename(columns={'o_id': 'id'})

    def pairs_of(classes):
        s = df[df.fake_cls.isin(classes)][cols].add_prefix('s_').rename(columns={'s_id': 'id'})
        p = osel.merge(s, on='id')
        for dm in ['FS', 'FA', 'TS', 'TA']:
            p[f'd_{dm}'] = p[f's_pmc_{dm}'] - p[f'o_pmc_{dm}']
        return p
    cap = pairs_of(['face_swap', 'face_attribute'])
    img = pairs_of(['text_swap', 'text_attribute'])
    cap['sib_cls'] = cap.s_fake_cls.map({'face_swap': 'FS', 'face_attribute': 'FA'})
    img['sib_cls'] = img.s_fake_cls.map({'text_swap': 'TS', 'text_attribute': 'TA'})
    cap.to_csv(f'{od}/pairs_caption_sib.csv', index=False)
    img.to_csv(f'{od}/pairs_image_sib.csv', index=False)

    # (a) 同句兄弟：文本维；(b) 同图兄弟：图像维
    ta = pd.concat([delta_table(cap, ['TS', 'TA'], 'sib_cls', B, rng),
                    delta_table(cap, ['TS', 'TA'], 's_method', B, rng).query("stratum != 'all'")], ignore_index=True)
    ta.to_csv(f'{od}/a_caption_sib_text_dims.csv', index=False)
    tb = delta_table(img, ['FS', 'FA'], 'sib_cls', B, rng)
    tb['n_pairs_image_path_differs'] = int((img.o_image != img.s_image).sum())
    tb.to_csv(f'{od}/b_image_sib_image_dims.csv', index=False)

    # (c) 配对排序：兄弟对应维 > 原图对应维
    cap['own_dim_higher'] = np.where(cap.sib_cls == 'FS', cap.s_pmc_FS > cap.o_pmc_FS, cap.s_pmc_FA > cap.o_pmc_FA).astype(float)
    rows = []
    for sname, sub in [('all', cap), ('orig_FP', cap[cap.o_fp == 1]), ('orig_TN', cap[cap.o_fp == 0])]:
        for strat, g in [('all', sub)] + [(f'method={k}', gg) for k, gg in sub.groupby('s_method')]:
            if len(g) == 0:
                continue
            lo, hi = cluster_boot(g, 'own_dim_higher', np.mean, B, rng)
            rows.append(dict(subset=sname, stratum=strat, n_pairs=len(g), n_ids=g.id.nunique(),
                             frac_sib_dim_higher=g.own_dim_higher.mean(), ci_lo=lo, ci_hi=hi, note=NOTE))
    pd.DataFrame(rows).to_csv(f'{od}/c_rank_caption_sib.csv', index=False)

    # (d) 覆盖率
    has_cap, has_img = set(cap.id), set(img.id)
    rows = []
    for sname, g in [('all', o), ('FP', o[o.fp == 1]), ('TN', o[o.fp == 0])]:
        rows.append(dict(split=a.split, subset=sname, n_orig=len(g), frac_has_caption_sib=g.id.isin(has_cap).mean(),
                         frac_has_image_sib=g.id.isin(has_img).mean(), frozen_sha256=sha))
    pd.DataFrame(rows).to_csv(f'{od}/d_coverage.csv', index=False)

    pd.set_option('display.width', 250); pd.set_option('display.max_columns', 30)
    print(pd.DataFrame(rows).round(4).to_string(index=False))
    print(f'[done] {od}  frozen_sha256={sha}  ({NOTE})')


if __name__ == '__main__':
    main(sys.argv[1:])
