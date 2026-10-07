"""全量 MAP / 多标签各类 F1，CPU 计算。用法: python check_multilabel_metrics.py <preds.pkl> [<preds.pkl> ...]

两种 F1 口径：
  A) 7/21 口径（复现 test.py 的 compute_per_category）：二分类判定在 fake_cls == 单一类 的子集上的 F1，与 prob_mc 无关；
  B) 多标签头口径：prob_mc[:, k] >= 0.5 vs target_mc[:, k]，全部样本。
MAP 复现 tools/multilabel_metrics.AveragePrecisionMeter.value()（difficult_examples=False）。
"""
import sys, pickle
import numpy as np

REF_721 = dict(MAP=84.2042, F1_FS=93.7879, F1_FA=95.9347, F1_TS=90.6481, F1_TA=91.5934)
DIMS = ['FS', 'FA', 'TS', 'TA']
CATS = ['face_swap', 'face_attribute', 'text_swap', 'text_attribute']


def average_precision(scores, targets):
    # 与 AveragePrecisionMeter.average_precision 等价：按分数降序，累加 prec@i
    order = np.argsort(-scores, kind='stable')
    t = targets[order] == 1
    pos_cum = np.cumsum(t)
    ranks = np.arange(1, len(t) + 1)
    return np.sum(pos_cum[t] / ranks[t]) / t.sum()


def f1(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp > 0 else 0.
    r = tp / (tp + fn) if tp + fn > 0 else 0.
    return 2 * p * r / (p + r) if p + r > 0 else 0.


def evaluate(path):
    rows = pickle.load(open(path, 'rb'))
    cls = np.array([r['fake_cls'] for r in rows])
    prob_fake = np.array([r['prob_fake'] for r in rows])
    pm = np.stack([r['prob_mc'] for r in rows]).astype(np.float64)
    tm = np.stack([r['target_mc'] for r in rows]).astype(int)
    y = (cls != 'orig').astype(int)
    pred_bin = (prob_fake >= 0.5).astype(int)

    out = {'n': len(rows)}
    ap = [average_precision(pm[:, k], tm[:, k]) for k in range(4)]
    out['MAP'] = np.mean(ap) * 100
    for k, d in enumerate(DIMS):
        out[f'AP_{d}'] = ap[k] * 100
    for d, c in zip(DIMS, CATS):
        m = cls == c
        yt, yp = y[m], pred_bin[m]
        out[f'F1_{d}'] = f1(np.sum((yt == 1) & (yp == 1)), np.sum((yt == 0) & (yp == 1)),
                            np.sum((yt == 1) & (yp == 0))) * 100
    for k, d in enumerate(DIMS):
        pk, tk = pm[:, k] >= 0.5, tm[:, k] == 1
        out[f'mlF1_{d}'] = f1(np.sum(pk & tk), np.sum(pk & ~tk), np.sum(~pk & tk)) * 100
        out[f'mlACC_{d}'] = np.mean(pk == tk) * 100
    return out


def main(paths):
    res = {p: evaluate(p) for p in paths}
    names = [p.rsplit('/', 1)[-1] for p in paths]
    print(f"{'metric':10s}" + ''.join(f"{n:>32s}" for n in names) + f"{'7/21':>10s}")
    keys = ['MAP'] + [f'AP_{d}' for d in DIMS] + [f'F1_{d}' for d in DIMS] \
        + [f'mlF1_{d}' for d in DIMS] + [f'mlACC_{d}' for d in DIMS]
    for k in keys:
        line = f"{k:10s}"
        for p in paths:
            v = res[p][k]
            line += f"{v:14.4f}" + (f" ({v - REF_721[k]:+9.4f})" if k in REF_721 else ' ' * 12) + ' ' * 6
        line += f"{REF_721[k]:10.4f}" if k in REF_721 else ''
        print(line)
    print("\nF1_* = 7/21 口径（二分类判定在单一类子集上）；mlF1_*/mlACC_* = 多标签头 prob_mc>=0.5，全部样本。括号内为与 7/21 之差。")


if __name__ == '__main__':
    main(sys.argv[1:])
