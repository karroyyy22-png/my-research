import pickle, collections
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
from scipy.optimize import brentq
from scipy.interpolate import interp1d

PKL = '/nas/data_2/wanhongz/failure_analysis/preds_test_all.pkl'
NPZ = '/nas/data_2/wanhongz/ASAP/results/log20260602_134300_train_4gpu_batch16/evaluation/raw_eval_data.npz'


def bin_metrics(y, p):
    auc = roc_auc_score(y, p)
    acc = np.mean((p >= 0.5).astype(int) == y)
    fpr, tpr, _ = roc_curve(y, p, pos_label=1)
    eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    return auc * 100, acc * 100, eer * 100


rows = pickle.load(open(PKL, 'rb'))
z = np.load(NPZ, allow_pickle=True)
mi = np.array([r['meta_index'] for r in rows])
assert len(mi) == 50705 and len(set(mi.tolist())) == 50705
cls = np.array([r['fake_cls'] for r in rows])
ours = np.array([r['prob_fake'] for r in rows])
ref = z['y_pred_asap'][mi]
y = (cls != 'orig').astype(int)
assert np.array_equal(y, z['y_true'][mi]), 'y_true 不一致'
assert np.array_equal(cls, z['label_all'][mi]), 'label 不一致'
assert all(r['image'] == p for r, p in zip(rows, z['img_path_all'][mi])), '路径不一致'

iou = np.array([r['iou_official'] for r in rows])
tp = sum(r['tok_tp'] for r in rows); fp = sum(r['tok_fp'] for r in rows); fn = sum(r['tok_fn'] for r in rows)
prec = tp / (tp + fp); rec = tp / (tp + fn); f1 = 2 * prec * rec / (prec + rec)

a, c, e = bin_metrics(y, ours)
ra, rc, re = bin_metrics(y, ref)
ref721 = dict(AUC=92.0633, ACC=85.5813, EER=15.2821, IoU=74.4318, IoU_ACC_50=81.6862, IoU_ACC_75=74.0400, F1_tok=70.3940)
mine = dict(AUC=a, ACC=c, EER=e, IoU=iou.mean() * 100, IoU_ACC_50=np.mean(iou > .5) * 100,
            IoU_ACC_75=np.mean(iou > .75) * 100, F1_tok=f1 * 100)
print(f"{'metric':12s} {'ours(pkl)':>10s} {'7/21':>10s} {'diff':>9s}")
for k in ref721:
    print(f"{k:12s} {mine[k]:10.4f} {ref721[k]:10.4f} {mine[k] - ref721[k]:+9.4f}")
print(f"\n[npz y_pred_asap] AUC={ra:.4f} ACC={rc:.4f} EER={re:.4f}")
print(f"tok TP={tp} FP={fp} FN={fn}  P={prec*100:.4f} R={rec*100:.4f}")

d = np.abs(ours - ref)
flip = (ours >= .5) != (ref >= .5)
print(f"\n|Δprob| max={d.max():.4e} mean={d.mean():.4e}  >1e-3: {np.sum(d > 1e-3)}  >1e-2: {np.sum(d > 1e-2)}")
print(f"判定翻转 {flip.sum()} / {len(flip)}  ({flip.mean()*100:.3f}%)")
order = sorted(set(cls.tolist()))
print(f"\n{'fake_cls':28s} {'n':>6s} {'flip':>5s} {'ours对':>6s} {'npz对':>6s} {'|Δ|>1e-2':>9s}")
for k in order:
    m = cls == k
    fm = m & flip
    ours_ok = np.sum(((ours >= .5).astype(int) == y) & fm)
    ref_ok = np.sum(((ref >= .5).astype(int) == y) & fm)
    print(f"{k:28s} {m.sum():6d} {fm.sum():5d} {ours_ok:6d} {ref_ok:6d} {np.sum(m & (d > 1e-2)):9d}")

print("\nTop10 |Δprob|:")
print(f"{'meta_index':>10s} {'fake_cls':28s} {'ours':>9s} {'npz':>9s} {'|Δ|':>8s} flip")
for i in np.argsort(-d)[:10]:
    print(f"{mi[i]:10d} {cls[i]:28s} {ours[i]:9.5f} {ref[i]:9.5f} {d[i]:8.5f} {bool(flip[i])}")
