"""B0：val/test 的 Original 在 train metadata 中是否有重复/同源样本（只读 metadata 与 pkl，不读图）。
用法: python b0_train_overlap.py --split val|test [--n 500]
输出: results/B0/<split>/ 下 per_orig.csv、summary.csv、hit_labels.csv
定义见 frozen_hypotheses.json 的 B0 段；只做描述，不下结论。
"""
import os, sys, json, argparse
from collections import Counter, defaultdict
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from b_common import check_frozen, load_split, cls6, boot_diff_props, out_dir  # noqa: E402

CHECKS = ['same_id', 'same_image', 'same_folder', 'same_caption']


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', choices=['val', 'test'], default='val')
    ap.add_argument('--n', type=int, default=0, help='小样本：随机抽 n 条 Original（0=全部）')
    a = ap.parse_args(argv)
    cfg, sha = check_frozen(a.split)
    seed, B = cfg['seed'], cfg['bootstrap']['B']
    od = out_dir('B0', a.split, a.n > 0)

    df, _ = load_split(cfg, a.split)
    o = df[df.fake_cls == 'orig'].copy()
    if a.n:
        o = o.sample(n=min(a.n, len(o)), random_state=seed)
    o['fp'] = (o.prob_fake >= 0.5).astype(int)

    train = json.load(open(os.path.join(cfg['inputs']['meta_dir'], 'train.json')))
    idx = {k: defaultdict(Counter) for k in CHECKS}
    for m in train:
        c = cls6(m['fake_cls'])
        idx['same_id'][m['id']][c] += 1
        idx['same_image'][m['image']][c] += 1
        if '/origin/' in m['image']:
            idx['same_folder'][m['image'].rsplit('/', 1)[0]][c] += 1
        idx['same_caption'][m['text'].strip()][c] += 1

    keys = {'same_id': o.id, 'same_image': o.image, 'same_folder': o.folder, 'same_caption': o.text.str.strip()}
    for k in CHECKS:
        cnts = [idx[k].get(v, Counter()) for v in keys[k]]
        o[f'{k}_hit'] = [int(sum(c.values()) > 0) for c in cnts]
        o[f'{k}_n'] = [sum(c.values()) for c in cnts]
        o[f'{k}_labels'] = [json.dumps(dict(c), sort_keys=True) for c in cnts]
        # 命中样本中被篡改（非 orig）的比例
        o[f'{k}_frac_manip'] = [(sum(c.values()) - c['orig']) / sum(c.values()) if sum(c.values()) else np.nan
                                for c in cnts]

    cols = ['meta_index', 'id', 'image', 'source', 'prob_fake', 'fp'] + \
        [f'{k}_{s}' for k in CHECKS for s in ('hit', 'n', 'labels', 'frac_manip')]
    o[cols].to_csv(f'{od}/per_orig.csv', index=False)

    rng = np.random.default_rng(seed)
    rows = []
    for stratum, g in [('all', o)] + [(s, gg) for s, gg in o.groupby('source')]:
        fp, tn = g[g.fp == 1], g[g.fp == 0]
        for k in CHECKS:
            lo, hi = boot_diff_props(fp[f'{k}_hit'], tn[f'{k}_hit'], B, rng)
            hit = g[g[f'{k}_hit'] == 1]
            rows.append(dict(split=a.split, stratum=stratum, check=k, n=len(g), n_fp=len(fp), n_tn=len(tn),
                             hit_rate=g[f'{k}_hit'].mean(), hit_rate_fp=fp[f'{k}_hit'].mean(),
                             hit_rate_tn=tn[f'{k}_hit'].mean(),
                             diff_fp_minus_tn=fp[f'{k}_hit'].mean() - tn[f'{k}_hit'].mean(),
                             diff_ci_lo=lo, diff_ci_hi=hi,
                             frac_manip_among_hits_fp=fp.loc[fp[f'{k}_hit'] == 1, f'{k}_frac_manip'].mean(),
                             frac_manip_among_hits_tn=tn.loc[tn[f'{k}_hit'] == 1, f'{k}_frac_manip'].mean(),
                             n_hit=len(hit)))
    summ = pd.DataFrame(rows)
    summ.to_csv(f'{od}/summary.csv', index=False)

    lab = []
    for k in CHECKS:
        for grp, g in (('FP', o[o.fp == 1]), ('TN', o[o.fp == 0])):
            tot = Counter()
            for s in g[f'{k}_labels']:
                tot.update(json.loads(s))
            n = sum(tot.values())
            lab.append(dict(split=a.split, check=k, group=grp, n_train_hits=n,
                            **{c: tot[c] / n if n else np.nan for c in ['orig', 'FS', 'FA', 'TS', 'TA', 'mixed']}))
    pd.DataFrame(lab).to_csv(f'{od}/hit_labels.csv', index=False)

    pd.set_option('display.width', 250); pd.set_option('display.max_columns', 30)
    print(summ[summ.stratum == 'all'].round(4).to_string(index=False))
    print(pd.DataFrame(lab).round(4).to_string(index=False))
    print(f'[out] {od}  frozen_sha256={sha}')


if __name__ == '__main__':
    main(sys.argv[1:])
