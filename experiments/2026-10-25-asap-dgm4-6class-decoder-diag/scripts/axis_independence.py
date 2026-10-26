"""E02 补充：图像轴/文本轴拆分（只读 E01_*_preds9.csv.gz，纯 CPU）。
用法: python axis_independence.py <results/E01 目录> <输出目录>
输出: E02_axis_independence.csv（Mixed 内单轴召回 vs 单类、联合/乘积比值，含 bootstrap 95% CI）
      E02_axis_error_composition.csv（每条轴的误报/漏检/类型混淆）
九类索引顺序（同 E01 preds9 的 y9）: orig,FS,FA,TS,TA,FS&TS,FS&TA,FA&TS,FA&TA
"""
import sys, os, numpy as np, pandas as pd
src, out = sys.argv[1], sys.argv[2]; os.makedirs(out, exist_ok=True)
IMG = np.array([0,1,2,0,0,1,1,2,2]); TXT = np.array([0,0,0,1,2,1,2,1,2])   # 图像:0无/1FS/2FA 文本:0无/1TS/2TA
PAIRS = [('FS&TS',5,'FS',1,'TS',3),('FS&TA',6,'FS',1,'TA',4),('FA&TS',7,'FA',2,'TS',3),('FA&TA',8,'FA',2,'TA',4)]
rng = np.random.RandomState(2026); B = 1000
ci = lambda v: (float(np.percentile(v,2.5)), float(np.percentile(v,97.5)))
rows, comp = [], []
for split in ['val_holdout','test']:
    d = pd.read_csv(os.path.join(src, f'E01_{split}_preds9.csv.gz')); y = d.y9.values
    for dec in ['A_tuned','B_tuned']:
        p = d[dec].values; ic = IMG[p]==IMG[y]; tc = TXT[p]==TXT[y]; both = ic & tc
        for nm, m, si, sic, st, stc in PAIRS + [('pooled_mixed', None, '', None, '', None)]:
            idx = np.where(np.isin(y,[5,6,7,8]) if m is None else y==m)[0]
            a, b, j = ic[idx].mean(), tc[idx].mean(), both[idx].mean()
            rs, da, db = [], [], []
            for _ in range(B):
                bm = rng.choice(idx, len(idx)); a_, b_ = ic[bm].mean(), tc[bm].mean()
                rs.append(both[bm].mean()/(a_*b_))
                if m is not None:
                    da.append(a_-ic[rng.choice(np.where(y==sic)[0], (y==sic).sum())].mean())
                    db.append(b_-tc[rng.choice(np.where(y==stc)[0], (y==stc).sum())].mean())
            r = dict(split=split, decoder=dec, group=nm, n=len(idx), img_recall=a, txt_recall=b, joint=j,
                     joint_over_product=j/(a*b), ratio_ci_lo=ci(rs)[0], ratio_ci_hi=ci(rs)[1])
            if m is not None:
                r.update(img_recall_single=ic[y==sic].mean(), img_diff_ci_lo=ci(da)[0], img_diff_ci_hi=ci(da)[1],
                         txt_recall_single=tc[y==stc].mean(), txt_diff_ci_lo=ci(db)[0], txt_diff_ci_hi=ci(db)[1])
            rows.append(r)
        for nm, ax in [('image_axis',IMG),('text_axis',TXT)]:
            ty, pr = ax[y], ax[p]; n0, n1 = (ty==0).sum(), (ty!=0).sum()
            fa = ((ty==0)&(pr!=0)).sum(); ms = ((ty!=0)&(pr==0)).sum(); tp = ((ty!=0)&(pr!=0)&(ty!=pr)).sum()
            comp.append(dict(split=split, decoder=dec, axis=nm, n_err=fa+ms+tp, err_rate=(fa+ms+tp)/len(y),
                             false_alarm=fa, false_alarm_rate=fa/n0, miss=ms, miss_rate=ms/n1, type_confusion=tp, type_confusion_rate=tp/n1))
pd.DataFrame(rows).to_csv(os.path.join(out,'E02_axis_independence.csv'), index=False)
pd.DataFrame(comp).to_csv(os.path.join(out,'E02_axis_error_composition.csv'), index=False)
print('done', len(rows), len(comp))
