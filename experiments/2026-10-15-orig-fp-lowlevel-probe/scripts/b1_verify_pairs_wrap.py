"""B1(f)：verify_pairs.py 的 V0 自然配对检验的包装版（不修改 verify_pairs.py；只读 pkl 与 metadata，不读图）。
与原脚本的差异（按 frozen_hypotheses.json 的 V0 段）：可选 --split；同句兄弟用完整 caption（strip 后完全相同，不截断）；
差值用兄弟簇 cluster bootstrap 给 95% CI 并按写死的判据判定；抽 20 对同句兄弟供人工核对。
P5/P9/P22/P29 只在 test 上作定性示例，本脚本不运行（需要时用原 verify_pairs.py）。
用法: python b1_verify_pairs_wrap.py --split val|test [--n 500]
"""
import os, sys, argparse
from collections import defaultdict
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from b_common import check_frozen, load_split, out_dir, DIMS  # noqa: E402

IMG_SIB = {'orig', 'text_swap', 'text_attribute'}
TXT_SIB = {'orig', 'face_swap', 'face_attribute'}


def cluster_boot_diff(fp, tn, B, rng):
    """fp/tn: DataFrame(cluster, rate)。以簇为单位分别重采样两组，返回均值差的 95% CI。"""
    def prep(d):
        g = {c: v.values for c, v in d.groupby('cluster').rate}
        return np.array(list(g)), g
    fk, fg = prep(fp); tk, tg = prep(tn)
    if len(fk) == 0 or len(tk) == 0:
        return np.nan, np.nan
    ds = []
    for _ in range(B):
        a = np.concatenate([fg[c] for c in fk[rng.integers(0, len(fk), len(fk))]])
        b = np.concatenate([tg[c] for c in tk[rng.integers(0, len(tk), len(tk))]])
        ds.append(a.mean() - b.mean())
    return float(np.percentile(ds, 2.5)), float(np.percentile(ds, 97.5))


def verdict(d, lo, hi):
    if np.isnan(d):
        return 'n/a'
    if d >= 0.05 and (lo > 0 or hi < 0):
        return '支持'
    if lo >= -0.02 and hi <= 0.02:
        return '不支持'
    return '证据不足'


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', choices=['val', 'test'], default='val')
    ap.add_argument('--n', type=int, default=0, help='小样本：只对随机 n 条 Original 计算（兄弟池仍为全 split）')
    a = ap.parse_args(argv)
    cfg, sha = check_frozen(a.split)
    seed, B = cfg['seed'], cfg['bootstrap']['B']
    rng = np.random.default_rng(seed)
    od = out_dir('B1_V0', a.split, a.n > 0)

    df, _ = load_split(cfg, a.split)
    df = df.reset_index(drop=True)
    df['cap_full'] = df.text.str.strip()
    pm = df[[f'pmc_{d}' for d in DIMS]].values
    df['img_alarm'] = ((pm[:, 0] >= .5) | (pm[:, 1] >= .5)).astype(int)
    df['txt_alarm'] = ((pm[:, 2] >= .5) | (pm[:, 3] >= .5)).astype(int)
    orig = df.fake_cls == 'orig'
    g = np.select([(df.img_alarm == 1) & (df.txt_alarm == 0), (df.img_alarm == 0) & (df.txt_alarm == 1),
                   (df.img_alarm == 1) & (df.txt_alarm == 1)], ['A', 'B', 'C'], default='D')
    df['grp'] = np.where(~orig, 'fake', np.where(df.prob_fake < .5, 'TN', g))

    by_img, by_cap = defaultdict(list), defaultdict(list)
    for i, (im, cp, fc) in enumerate(zip(df.image, df.cap_full, df.fake_cls)):
        if fc in IMG_SIB:
            by_img[im].append(i)
        if fc in TXT_SIB:
            by_cap[cp].append(i)

    targets = np.where(orig)[0]
    if a.n:
        targets = np.sort(rng.choice(targets, min(a.n, len(targets)), replace=False))
    recs = []
    for i in targets:
        si = [j for j in by_img[df.image.iat[i]] if j != i]
        st = [j for j in by_cap[df.cap_full.iat[i]] if j != i]
        recs.append(dict(meta_index=df.meta_index.iat[i], id=df.id.iat[i], grp=df.grp.iat[i], image=df.image.iat[i],
                         cap=df.cap_full.iat[i], n_img_sib=len(si), n_txt_sib=len(st),
                         img_sib_img_alarm=df.img_alarm.values[si].mean() if si else np.nan,
                         txt_sib_txt_alarm=df.txt_alarm.values[st].mean() if st else np.nan))
    sv = pd.DataFrame(recs)
    sv.drop(columns=['cap']).to_csv(f'{od}/V0_sibling_per_orig.csv', index=False)

    cov = sv.groupby('grp').agg(n=('meta_index', 'size'), has_img_sib=('n_img_sib', lambda x: (x > 0).mean()),
                                img_sib_img_alarm=('img_sib_img_alarm', 'mean'),
                                has_txt_sib=('n_txt_sib', lambda x: (x > 0).mean()),
                                txt_sib_txt_alarm=('txt_sib_txt_alarm', 'mean')).reindex(['A', 'B', 'C', 'D', 'TN'])
    cov.to_csv(f'{od}/V0_coverage_by_group.csv')

    rows = []
    # v4：以 id 为簇；B 组（同句兄弟 = 同一原图只改人脸）只作描述，不判定
    for test_grp, col in (('A', 'img_sib_img_alarm'), ('B', 'txt_sib_txt_alarm')):
        fp = sv[(sv.grp == test_grp) & sv[col].notna()].rename(columns={col: 'rate', 'id': 'cluster'})
        tn = sv[(sv.grp == 'TN') & sv[col].notna()].rename(columns={col: 'rate', 'id': 'cluster'})
        d = fp.rate.mean() - tn.rate.mean() if len(fp) and len(tn) else np.nan
        lo, hi = cluster_boot_diff(fp[['cluster', 'rate']], tn[['cluster', 'rate']], B, rng)
        rows.append(dict(split=a.split, test=f'{test_grp} vs TN: {col}', n_fp_with_sib=len(fp), n_tn_with_sib=len(tn),
                         rate_fp=fp.rate.mean(), rate_tn=tn.rate.mean(), diff=d, ci_lo=lo, ci_hi=hi,
                         verdict=verdict(d, lo, hi) if test_grp == 'A' else '仅描述（同句兄弟与原图只在人脸区域不同，'
                         '背景相同，文本维报警不能归因于 caption）', frozen_sha256=sha))
    res = pd.DataFrame(rows); res.to_csv(f'{od}/V0_tests.csv', index=False)

    # 人工核对：随机 20 对同句兄弟
    pairs = []
    for i in targets:
        st = [j for j in by_cap[df.cap_full.iat[i]] if j != i]
        if st:
            pairs.append((i, st[0]))
    pick = [pairs[k] for k in rng.choice(len(pairs), min(20, len(pairs)), replace=False)] if pairs else []
    pd.DataFrame([dict(a_meta=df.meta_index.iat[i], a_cls=df.fake_cls.iat[i], a_image=df.image.iat[i],
                       b_meta=df.meta_index.iat[j], b_cls=df.fake_cls.iat[j], b_image=df.image.iat[j],
                       caption=df.cap_full.iat[i]) for i, j in pick]).to_csv(f'{od}/V0_caption_pairs_manual20.csv', index=False)

    pd.set_option('display.width', 250); pd.set_option('display.max_columns', 30)
    print(cov.round(4).to_string()); print(res.round(4).to_string(index=False))
    print(f'[done] {od}  frozen_sha256={sha}')


if __name__ == '__main__':
    main(sys.argv[1:])
