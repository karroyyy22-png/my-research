"""事后描述，不作判定：V0 的 A 组（图像维误报）vs TN，同图兄弟按 TS / TA 分层的图像维报警率。
不属于 frozen_hypotheses.json 的任何判据；只读 pkl 与 metadata，不读图；以 id 为簇 bootstrap。
用法: python posthoc_v0_A_by_sibcls.py --split val|test
输出: results/B1_V0/<split>/posthoc_A_by_sib_cls.csv
"""
import os, sys, argparse
from collections import defaultdict
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from b_common import check_frozen, load_split, out_dir, DIMS  # noqa: E402
from b1_verify_pairs_wrap import cluster_boot_diff  # noqa: E402

LABEL = '事后描述，不作判定'


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', choices=['val', 'test'], default='val')
    a = ap.parse_args(argv)
    cfg, sha = check_frozen(a.split)
    rng = np.random.default_rng(cfg['seed'])
    od = out_dir('B1_V0', a.split, False)

    df, _ = load_split(cfg, a.split)
    df = df.reset_index(drop=True)
    pm = df[[f'pmc_{d}' for d in DIMS]].values
    img_alarm = ((pm[:, 0] >= .5) | (pm[:, 1] >= .5)).astype(int)
    txt_alarm = (pm[:, 2] >= .5) | (pm[:, 3] >= .5)
    orig = (df.fake_cls == 'orig').values
    grp = np.where(~orig, 'fake', np.where(df.prob_fake < .5, 'TN',
                   np.where((img_alarm == 1) & ~txt_alarm, 'A', 'other_FP')))
    by_img = defaultdict(list)
    for i, (im, fc) in enumerate(zip(df.image, df.fake_cls)):
        if fc in ('text_swap', 'text_attribute'):
            by_img[im].append(i)

    rows = []
    for sib_cls in ['text_swap', 'text_attribute']:
        recs = []
        for i in np.where(orig & np.isin(grp, ['A', 'TN']))[0]:
            sib = [j for j in by_img[df.image.iat[i]] if df.fake_cls.iat[j] == sib_cls]
            if sib:
                recs.append(dict(grp=grp[i], cluster=df.id.iat[i], rate=img_alarm[sib].mean()))
        r = pd.DataFrame(recs)
        fp, tn = r[r.grp == 'A'], r[r.grp == 'TN']
        lo, hi = cluster_boot_diff(fp[['cluster', 'rate']], tn[['cluster', 'rate']], cfg['bootstrap']['B'], rng)
        rows.append(dict(split=a.split, sibling_class=sib_cls, n_A_with_sib=len(fp), n_TN_with_sib=len(tn),
                         rate_A=fp.rate.mean(), rate_TN=tn.rate.mean(), diff=fp.rate.mean() - tn.rate.mean(),
                         ci_lo=lo, ci_hi=hi, label=LABEL, frozen_sha256=sha))
    out = pd.DataFrame(rows)
    out.to_csv(f'{od}/posthoc_A_by_sib_cls.csv', index=False)
    pd.set_option('display.width', 250)
    print(out.drop(columns='frozen_sha256').round(4).to_string(index=False))


if __name__ == '__main__':
    main(sys.argv[1:])
