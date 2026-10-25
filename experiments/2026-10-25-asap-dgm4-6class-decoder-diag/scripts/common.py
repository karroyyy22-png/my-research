"""公共工具：加载逐样本预测、标签映射、解码器、指标。纯 CPU，只读 pkl + metadata。

加载思路沿用 experiments/2026-10-07-asap-failure-analysis/scripts/analyze_failures.py 的 load_preds / build_frame。
"""
import os, json, pickle
import numpy as np
import pandas as pd

FA_DIR = '/nas/data_2/wanhongz/failure_analysis'
META_DIR = '/nas/data_2/wanhongz/datasets/DGM4/metadata'
PRED_PATH = {'val': f'{FA_DIR}/preds_val_all.pkl', 'test': f'{FA_DIR}/preds_test_all.pkl'}  # 禁用 preUtilsC
EXP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES_DIR = os.path.join(EXP_DIR, 'results')

DIMS = ['FS', 'FA', 'TS', 'TA']
CLASSES9 = ['orig', 'face_swap', 'face_attribute', 'text_swap', 'text_attribute',
            'face_swap&text_swap', 'face_swap&text_attribute',
            'face_attribute&text_swap', 'face_attribute&text_attribute']
SHORT9 = ['orig', 'FS', 'FA', 'TS', 'TA', 'FS&TS', 'FS&TA', 'FA&TS', 'FA&TA']
CLASSES6 = ['Original', 'FS', 'FA', 'TS', 'TA', 'Mixed']
# 九类 = 图像状态{0:无,1:FS,2:FA} × 文本状态{0:无,1:TS,2:TA}
IMG_STATES, TXT_STATES = ['none', 'FS', 'FA'], ['none', 'TS', 'TA']
C9_AXES = {0: (0, 0), 1: (1, 0), 2: (2, 0), 3: (0, 1), 4: (0, 2), 5: (1, 1), 6: (1, 2), 7: (2, 1), 8: (2, 2)}
AXES_C9 = {v: k for k, v in C9_AXES.items()}
C9_TO_C6 = np.array([0, 1, 2, 3, 4, 5, 5, 5, 5])
SOURCES = ['bbc', 'guardian', 'usa_today', 'washington_post']
SIZE_EDGES = [0, 32, 64, 128, 256, np.inf]
SIZE_LABELS = ['<32', '32-64', '64-128', '128-256', '>=256']
PFAKE_EDGES = [0, .1, .3, .5, .7, .9, 1.0000001]
PFAKE_LABELS = ['[0,.1)', '[.1,.3)', '[.3,.5)', '[.5,.7)', '[.7,.9)', '[.9,1]']
EPS = 1e-6
SMALL_N = 200


def cls_to_mc(c):
    return np.array([int('face_swap' in c), int('face_attribute' in c),
                     int('text_swap' in c), int('text_attribute' in c)], dtype=np.int8)


def load_frame(split):
    rows = pickle.load(open(PRED_PATH[split], 'rb'))
    meta = json.load(open(f'{META_DIR}/{split}.json'))
    recs = []
    for r in rows:
        m = meta[r['meta_index']]
        mt = np.asarray(m['mtcnn_boxes'], dtype=np.float64).reshape(-1, 4)
        sz = np.sqrt((mt[:, 2] - mt[:, 0]) * (mt[:, 3] - mt[:, 1])) if len(mt) else np.array([np.nan])
        src = [s for s in SOURCES if f'/{s}/' in r['image']]
        recs.append(dict(
            meta_index=r['meta_index'], image=r['image'], fake_cls=r['fake_cls'],
            meta_image=m['image'], meta_fake_cls=m['fake_cls'],
            prob_fake=float(r['prob_fake']), pred_cls=int(r['pred_cls']),
            **{f'p_{d}': float(r['prob_mc'][k]) for k, d in enumerate(DIMS)},
            **{f't_{d}': int(r['target_mc'][k]) for k, d in enumerate(DIMS)},
            source=src[0] if len(src) == 1 else 'unknown',
            max_face=float(np.max(sz)),
        ))
    df = pd.DataFrame(recs)
    df['y9'] = df.fake_cls.map({c: i for i, c in enumerate(CLASSES9)}).astype(int)
    df['y6'] = C9_TO_C6[df.y9.values]
    df['face_bin'] = pd.cut(df.max_face, SIZE_EDGES, right=False, labels=SIZE_LABELS).astype(str)
    df['pfake_bin'] = pd.cut(df.prob_fake, PFAKE_EDGES, right=False, labels=PFAKE_LABELS).astype(str)
    return df


def probs(df):
    """返回 (p_fake[n], P[n,4])，四维顺序 FS,FA,TS,TA。"""
    return df.prob_fake.values.astype(np.float64), df[[f'p_{d}' for d in DIMS]].values.astype(np.float64)


# ---------------------------------------------------------------- 解码器
def decode_A(pf, P, t_b, t_m):
    """Binary-first。返回九类索引。"""
    n = len(pf)
    img = np.where(np.maximum(P[:, 0], P[:, 1]) >= t_m, np.where(P[:, 0] >= P[:, 1], 1, 2), 0)
    txt = np.where(np.maximum(P[:, 2], P[:, 3]) >= t_m, np.where(P[:, 2] >= P[:, 3], 1, 2), 0)
    # fallback：Binary 判假但两侧都"无" → 四维 argmax，只设置所属模态
    fb = (img == 0) & (txt == 0)
    am = np.argmax(P, axis=1)
    img = np.where(fb & (am < 2), am + 1, img)
    txt = np.where(fb & (am >= 2), am - 1, txt)
    y = np.array([AXES_C9[(i, t)] for i, t in zip(img, txt)], dtype=int)
    y[pf < t_b] = 0
    return y


def scores_B(P):
    """九类联合得分（未加偏置），[n,9]。"""
    Q = np.clip(P, EPS, 1 - EPS)
    l1, l0 = np.log(Q), np.log(1 - Q)
    s_img = np.stack([l0[:, 0] + l0[:, 1], l1[:, 0] + l0[:, 1], l0[:, 0] + l1[:, 1]], 1)
    s_txt = np.stack([l0[:, 2] + l0[:, 3], l1[:, 2] + l0[:, 3], l0[:, 2] + l1[:, 3]], 1)
    S = np.zeros((len(P), 9))
    for c, (i, t) in C9_AXES.items():
        S[:, c] = s_img[:, i] + s_txt[:, t]
    return S


def decode_B(P, b_orig, S=None):
    S = scores_B(P) if S is None else S
    S = S.copy()
    S[:, 0] += b_orig
    return np.argmax(S, axis=1)


# ---------------------------------------------------------------- 指标
def confusion(y, p, k):
    return np.bincount(y * k + p, minlength=k * k).reshape(k, k)


def prf_from_cm(cm):
    tp = np.diag(cm).astype(float)
    fp = cm.sum(0) - tp
    fn = cm.sum(1) - tp
    with np.errstate(invalid='ignore', divide='ignore'):
        P = np.where(tp + fp > 0, tp / (tp + fp), 0.)
        R = np.where(tp + fn > 0, tp / (tp + fn), np.nan)
        F = np.where(P + R > 0, 2 * P * R / (P + R), 0.)
    return P, R, F


def macro_f1_6(y6, p6):
    _, _, F = prf_from_cm(confusion(y6, p6, 6))
    return float(np.mean(F))


def summary_metrics(y9, p9):
    """六类/九类主要指标 + 由九类解码得到的四维多标签指标。"""
    y6, p6 = C9_TO_C6[y9], C9_TO_C6[p9]
    cm6, cm9 = confusion(y6, p6, 6), confusion(y9, p9, 9)
    P6, R6, F6 = prf_from_cm(cm6)
    P9, R9, F9 = prf_from_cm(cm9)
    out = dict(macroF1_6=float(np.mean(F6)), balAcc_6=float(np.nanmean(R6)), acc_6=float((y6 == p6).mean()),
               macroF1_9=float(np.mean(F9)), acc_9=float((y9 == p9).mean()))
    for i, c in enumerate(CLASSES6):
        out[f'6_{c}_P'], out[f'6_{c}_R'], out[f'6_{c}_F1'] = P6[i], R6[i], F6[i]
    for i, c in enumerate(SHORT9):
        out[f'9_{c}_P'], out[f'9_{c}_R'], out[f'9_{c}_F1'] = P9[i], R9[i], F9[i]
    Ty = np.stack([cls9_mc(y9)[:, k] for k in range(4)], 1)
    Tp = cls9_mc(p9)
    for k, d in enumerate(DIMS):
        tp = int(((Ty[:, k] == 1) & (Tp[:, k] == 1)).sum()); fp = int(((Ty[:, k] == 0) & (Tp[:, k] == 1)).sum())
        fn = int(((Ty[:, k] == 1) & (Tp[:, k] == 0)).sum())
        out[f'ml_{d}_F1'] = 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else np.nan
        out[f'ml_{d}_R'] = tp / (tp + fn) if tp + fn else np.nan
    out['ml_exact_match'] = float(np.all(Ty == Tp, axis=1).mean())
    orig = y9 == 0
    out['orig_FPR'] = float((p9[orig] != 0).mean())  # Original 被判为任何篡改类的比例
    out['orig_R'] = 1 - out['orig_FPR']
    return out


_MC9 = np.array([[int(i == 1), int(i == 2), int(t == 1), int(t == 2)] for c, (i, t) in sorted(C9_AXES.items())])


def cls9_mc(y9):
    return _MC9[y9]


def binary_metrics(y_fake, pred_fake):
    tp = int((y_fake & pred_fake).sum()); fp = int((~y_fake & pred_fake).sum())
    fn = int((y_fake & ~pred_fake).sum()); tn = int((~y_fake & ~pred_fake).sum())
    return dict(bin_acc=(tp + tn) / len(y_fake), bin_P=tp / (tp + fp) if tp + fp else np.nan,
                bin_R=tp / (tp + fn), bin_F1=2 * tp / (2 * tp + fp + fn), orig_FP=fp, orig_FPR=fp / (fp + tn))


def multilabel_head_metrics(P, T, thr=0.5):
    """原始四个多标签头（各自阈值 thr）的 F1 与 Exact Match。"""
    out = {}
    pr = P >= thr
    for k, d in enumerate(DIMS):
        t = T[:, k] == 1
        tp, fp, fn = int((pr[:, k] & t).sum()), int((pr[:, k] & ~t).sum()), int((~pr[:, k] & t).sum())
        out[f'head_{d}_F1'] = 2 * tp / (2 * tp + fp + fn)
    out['head_exact_match'] = float(np.all(pr == (T == 1), axis=1).mean())
    return out


def md_table(df, fmt='{:.4f}'):
    cols = list(df.columns)
    lines = ['| ' + ' | '.join(map(str, cols)) + ' |', '|' + '---|' * len(cols)]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if isinstance(v, (float, np.floating)):
                cells.append('' if np.isnan(v) else fmt.format(v))
            else:
                cells.append(str(v))
        lines.append('| ' + ' | '.join(cells) + ' |')
    return '\n'.join(lines)
