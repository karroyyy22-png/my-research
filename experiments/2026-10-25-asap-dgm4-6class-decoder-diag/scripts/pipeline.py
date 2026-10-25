"""E01/E02/E03 的拟合（只用 val_dev）、预测与逐 split 报告。

fit_all(dev)            -> 冻结配置 dict（所有阈值/偏置/校准器参数都只来自 val_dev）
predict_all(cfg, df)    -> {解码器名: 九类预测}，以及二分类参考线
report_split(cfg, df, split) -> 写 results/E0x/*，返回汇总 dict
"""
import os, json
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.linear_model import LogisticRegression
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from common import *

GRID_T = np.round(np.arange(0.02, 0.981, 0.01), 2)
GRID_B = np.round(np.arange(-10, 20.001, 0.05), 2)
SWEEP_B = np.round(np.arange(-10, 20.001, 0.5), 2)
SWEEP_TB = np.round(np.arange(0.02, 0.981, 0.02), 2)
R1_THR = 0.736491
LR_C = 1.0
N_BOOT, BOOT_SEED = 1000, 2026
DECODERS = ['A_untuned', 'A_tuned', 'A_matchB', 'B_b0', 'B_tuned', 'B_matchA', 'B_TS_b0', 'B_TS_tuned', 'C_LR']
PAIRS = [('A_tuned', 'A_untuned'), ('B_tuned', 'B_b0'), ('B_tuned', 'A_tuned'),
         ('B_matchA', 'A_tuned'), ('B_tuned', 'A_matchB'),
         ('B_TS_b0', 'B_b0'), ('B_TS_tuned', 'B_tuned'), ('C_LR', 'A_tuned'), ('C_LR', 'B_tuned')]
BOOT_METRICS = ['macroF1_6', 'balAcc_6', 'acc_6', 'macroF1_9', 'orig_R']
E02_DECODERS = ['A_tuned', 'B_tuned']

INK, INK2, MUTED, GRID = '#0b0b0b', '#52514e', '#898781', '#e1e0d9'
SERIES = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#6250d6', '#e34948']
SEQ = LinearSegmentedColormap.from_list('seq', ['#fcfcfb', '#cde2fb', '#86b6ef', '#3987e5', '#1c5cab', '#0d366b'])


def logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p) - np.log(1 - p)


def sig(z):
    return 1 / (1 + np.exp(-z))


# ---------------------------------------------------------------- A 的快速网格
def _A_states(P, t_m):
    img = np.where(np.maximum(P[:, 0], P[:, 1]) >= t_m, np.where(P[:, 0] >= P[:, 1], 1, 2), 0)
    txt = np.where(np.maximum(P[:, 2], P[:, 3]) >= t_m, np.where(P[:, 2] >= P[:, 3], 1, 2), 0)
    fb = (img == 0) & (txt == 0)
    am = np.argmax(P, axis=1)
    img = np.where(fb & (am < 2), am + 1, img)
    txt = np.where(fb & (am >= 2), am - 1, txt)
    lut = np.array([AXES_C9[(i, t)] for i in range(3) for t in range(3)])
    return lut[img * 3 + txt]


def orig_recall(y9, p9):
    return float((p9[y9 == 0] == 0).mean())


def nearest(cands, key, ref):
    """cands: list of (score, param...)；取 score 最大，并列取 key 距离 ref 最近。"""
    best = max(c[0] for c in cands)
    tie = [c for c in cands if abs(c[0] - best) <= 1e-12]
    return min(tie, key=lambda c: key(c, ref))


def fit_A(pf, P, y9):
    y6 = C9_TO_C6[y9]
    cands = []
    for tm in GRID_T:
        base = _A_states(P, tm)
        for tb in GRID_T:
            p9 = np.where(pf < tb, 0, base)
            cands.append((macro_f1_6(y6, C9_TO_C6[p9]), tb, tm))
    s, tb, tm = nearest(cands, lambda c, r: (c[1] - .5) ** 2 + (c[2] - .5) ** 2, None)
    return dict(t_b=float(tb), t_m=float(tm), dev_macroF1_6=float(s))


def fit_B(P, y9):
    S = scores_B(P)
    y6 = C9_TO_C6[y9]
    cands = [(macro_f1_6(y6, C9_TO_C6[decode_B(P, b, S)]), b) for b in GRID_B]
    s, b = nearest(cands, lambda c, r: abs(c[1]), None)
    return dict(b_orig=float(b), dev_macroF1_6=float(s))


def match_B_to_recall(P, y9, target):
    S = scores_B(P)
    cands = [(-abs(orig_recall(y9, decode_B(P, b, S)) - target), b) for b in GRID_B]
    s, b = nearest(cands, lambda c, r: abs(c[1]), None)
    return dict(b_orig=float(b), target_orig_R=target, dev_orig_R=orig_recall(y9, decode_B(P, b, S)))


def match_A_to_recall(pf, P, y9, t_m, target):
    base = _A_states(P, t_m)
    cands = [(-abs(orig_recall(y9, np.where(pf < tb, 0, base)) - target), tb) for tb in np.round(np.arange(0.001, 0.9995, 0.001), 3)]
    s, tb = nearest(cands, lambda c, r: abs(c[1] - .5), None)
    return dict(t_b=float(tb), t_m=float(t_m), target_orig_R=target,
                dev_orig_R=orig_recall(y9, np.where(pf < tb, 0, base)))


def fit_temps(P, T):
    temps = []
    for k in range(4):
        z, t = logit(P[:, k]), T[:, k]
        def nll(logT):
            q = np.clip(sig(z / np.exp(logT)), 1e-12, 1 - 1e-12)
            return -np.mean(t * np.log(q) + (1 - t) * np.log(1 - q))
        r = minimize_scalar(nll, bounds=(-3, 3), method='bounded')
        temps.append(float(np.exp(r.x)))
    return temps


def apply_temps(P, temps):
    return np.stack([sig(logit(P[:, k]) / temps[k]) for k in range(4)], 1)


def lr_features(pf, P, mu=None, sd=None):
    X = np.column_stack([logit(pf)] + [logit(P[:, k]) for k in range(4)])
    if mu is None:
        mu, sd = X.mean(0), X.std(0)
    return (X - mu) / sd, mu, sd


def fit_all(dev):
    pf, P = probs(dev)
    y9 = dev.y9.values
    T = dev[[f't_{d}' for d in DIMS]].values
    cfg = dict(R0=dict(t=0.5), R1=dict(t=R1_THR), A_untuned=dict(t_b=0.5, t_m=0.5), B_b0=dict(b_orig=0.0))
    cfg['A_tuned'] = fit_A(pf, P, y9)
    cfg['B_tuned'] = fit_B(P, y9)
    rA = orig_recall(y9, decode_A(pf, P, cfg['A_tuned']['t_b'], cfg['A_tuned']['t_m']))
    rB = orig_recall(y9, decode_B(P, cfg['B_tuned']['b_orig']))
    cfg['B_matchA'] = match_B_to_recall(P, y9, rA)
    cfg['A_matchB'] = match_A_to_recall(pf, P, y9, cfg['A_tuned']['t_m'], rB)
    temps = fit_temps(P, T)
    cfg['B_TS_b0'] = dict(temps=temps, b_orig=0.0)
    cfg['B_TS_tuned'] = dict(temps=temps, **fit_B(apply_temps(P, temps), y9))
    X, mu, sd = lr_features(pf, P)
    lr = LogisticRegression(C=LR_C, max_iter=5000)  # lbfgs + 多类 = multinomial
    lr.fit(X, y9)
    cfg['C_LR'] = dict(C=LR_C, penalty='l2', target='九类', features=['logit(p_fake)', 'logit(p_FS)', 'logit(p_FA)', 'logit(p_TS)', 'logit(p_TA)'],
                       standardize_mu=mu.tolist(), standardize_sd=sd.tolist(),
                       classes=lr.classes_.tolist(), coef=lr.coef_.tolist(), intercept=lr.intercept_.tolist())
    return cfg


def predict_all(cfg, df):
    pf, P = probs(df)
    out = {
        'A_untuned': decode_A(pf, P, **{k: cfg['A_untuned'][k] for k in ['t_b', 't_m']}),
        'A_tuned': decode_A(pf, P, cfg['A_tuned']['t_b'], cfg['A_tuned']['t_m']),
        'A_matchB': decode_A(pf, P, cfg['A_matchB']['t_b'], cfg['A_matchB']['t_m']),
        'B_b0': decode_B(P, cfg['B_b0']['b_orig']),
        'B_tuned': decode_B(P, cfg['B_tuned']['b_orig']),
        'B_matchA': decode_B(P, cfg['B_matchA']['b_orig']),
    }
    Pt = apply_temps(P, cfg['B_TS_b0']['temps'])
    out['B_TS_b0'] = decode_B(Pt, cfg['B_TS_b0']['b_orig'])
    out['B_TS_tuned'] = decode_B(Pt, cfg['B_TS_tuned']['b_orig'])
    c = cfg['C_LR']
    X, _, _ = lr_features(pf, P, np.array(c['standardize_mu']), np.array(c['standardize_sd']))
    logits = X @ np.array(c['coef']).T + np.array(c['intercept'])
    out['C_LR'] = np.array(c['classes'])[np.argmax(logits, 1)]
    return out


# ---------------------------------------------------------------- bootstrap
def boot_metrics_from_cm(cm6, cm9):
    P6, R6, F6 = prf_from_cm(cm6)
    _, _, F9 = prf_from_cm(cm9)
    tot = cm6.sum()
    return dict(macroF1_6=F6.mean(), balAcc_6=np.nanmean(R6), acc_6=np.trace(cm6) / tot,
                macroF1_9=F9.mean(), orig_R=cm6[0, 0] / cm6[0].sum())


def paired_bootstrap(y9, preds, pairs):
    rng = np.random.RandomState(BOOT_SEED)
    n = len(y9)
    y6 = C9_TO_C6[y9]
    names = sorted({x for p in pairs for x in p})
    idx9 = {k: y9 * 9 + preds[k] for k in names}
    idx6 = {k: y6 * 6 + C9_TO_C6[preds[k]] for k in names}
    point = {k: boot_metrics_from_cm(np.bincount(idx6[k], minlength=36).reshape(6, 6),
                                     np.bincount(idx9[k], minlength=81).reshape(9, 9)) for k in names}
    samples = {k: {m: np.empty(N_BOOT) for m in BOOT_METRICS} for k in names}
    for b in range(N_BOOT):
        w = rng.multinomial(n, np.full(n, 1 / n)).astype(float)
        for k in names:
            r = boot_metrics_from_cm(np.bincount(idx6[k], weights=w, minlength=36).reshape(6, 6),
                                     np.bincount(idx9[k], weights=w, minlength=81).reshape(9, 9))
            for m in BOOT_METRICS:
                samples[k][m][b] = r[m]
    rows = []
    for a, c in pairs:
        for m in BOOT_METRICS:
            d = samples[a][m] - samples[c][m]
            lo, hi = np.percentile(d, [2.5, 97.5])
            rows.append(dict(pair=f'{a} - {c}', metric=m, diff=point[a][m] - point[c][m], ci95_lo=lo, ci95_hi=hi,
                             ci_excludes_0=bool(lo > 0 or hi < 0)))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- 画图
def _style(ax):
    for s in ['top', 'right']:
        ax.spines[s].set_visible(False)
    for s in ['left', 'bottom']:
        ax.spines[s].set_color('#c3c2b7')
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def heatmap(cm, labels, title, path):
    rn = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    k = len(labels)
    fig, axes = plt.subplots(1, 2, figsize=(2 * (0.75 * k + 1.6), 0.7 * k + 1.4), facecolor='#fcfcfb')
    for ax, M, sub in [(axes[0], cm, 'count'), (axes[1], rn, 'row-normalized')]:
        ax.imshow(rn, cmap=SEQ, vmin=0, vmax=1)
        for i in range(k):
            for j in range(k):
                txt = f'{int(M[i, j])}' if sub == 'count' else f'{M[i, j]:.2f}'
                ax.text(j, i, txt, ha='center', va='center', fontsize=7, color='#fcfcfb' if rn[i, j] > 0.55 else INK)
        ax.set_xticks(range(k)); ax.set_yticks(range(k))
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8, color=INK2)
        ax.set_yticklabels(labels, fontsize=8, color=INK2)
        ax.set_xlabel('predicted', fontsize=8, color=MUTED); ax.set_ylabel('true', fontsize=8, color=MUTED)
        ax.set_title(f'{title} ({sub})', fontsize=9, color=INK, loc='left')
        for s in ax.spines.values():
            s.set_visible(False)
        ax.tick_params(length=0)
    fig.tight_layout()
    fig.savefig(path, dpi=130, facecolor='#fcfcfb')
    plt.close(fig)


def sweep_plot(sw, title, path):
    """sw: 含 decoder/orig_R/R_FS.. 的 DataFrame；每个子图一种类型的 Recall 对 Original Recall。"""
    panels = ['R_FS', 'R_FA', 'R_TS', 'R_TA', 'R_Mixed', 'macroF1_6']
    fig, axes = plt.subplots(2, 3, figsize=(11, 6.4), facecolor='#fcfcfb', sharex=True)
    for ax, col in zip(axes.ravel(), panels):
        for i, (dec, g) in enumerate(sw.groupby('decoder', sort=False)):
            g = g.sort_values('orig_R')
            ax.plot(g.orig_R, g[col], lw=2, color=SERIES[i], label=dec, marker='o', ms=3)
        _style(ax)
        ax.set_title(col.replace('R_', 'Recall ') if col.startswith('R_') else 'Macro-F1 (6-class)', fontsize=9, color=INK, loc='left')
        ax.set_ylim(0, 1)
    for ax in axes[1]:
        ax.set_xlabel('Original Recall', fontsize=8, color=MUTED)
    axes[0, 0].legend(fontsize=8, frameon=False)
    fig.suptitle(title, fontsize=10, color=INK, x=0.01, ha='left')
    fig.tight_layout()
    fig.savefig(path, dpi=130, facecolor='#fcfcfb')
    plt.close(fig)


# ---------------------------------------------------------------- E01
def recall6(y9, p9):
    y6, p6 = C9_TO_C6[y9], C9_TO_C6[p9]
    _, R, F = prf_from_cm(confusion(y6, p6, 6))
    return dict(orig_R=R[0], **{f'R_{c}': R[i] for i, c in enumerate(CLASSES6) if i}, macroF1_6=F.mean())


def bias_sweep(cfg, df):
    pf, P = probs(df)
    y9 = df.y9.values
    S = scores_B(P)
    rows = [dict(decoder='B (sweep b_orig)', param=b, **recall6(y9, decode_B(P, b, S))) for b in SWEEP_B]
    base = _A_states(P, cfg['A_tuned']['t_m'])
    rows += [dict(decoder=f"A (sweep t_b, t_m={cfg['A_tuned']['t_m']})", param=tb, **recall6(y9, np.where(pf < tb, 0, base)))
             for tb in SWEEP_TB]
    return pd.DataFrame(rows)


def e01(cfg, df, split, preds, out):
    pf, P = probs(df)
    y9 = df.y9.values
    yb = y9 != 0
    rows = []
    for r in ['R0', 'R1']:
        rows.append(dict(decoder=r, **binary_metrics(yb, pf >= cfg[r]['t'])))
    bin_tab = pd.DataFrame(rows)
    bin_tab.to_csv(f'{out}/E01_{split}_binary_refs.csv', index=False)
    T = df[[f't_{d}' for d in DIMS]].values
    heads = multilabel_head_metrics(P, T)
    pd.DataFrame([heads]).to_csv(f'{out}/E01_{split}_raw_heads_thr0.5.csv', index=False)
    rows = []
    for k in DECODERS:
        m = summary_metrics(y9, preds[k])
        m.update(binary_metrics(yb, preds[k] != 0))
        rows.append(dict(decoder=k, **m))
    full = pd.DataFrame(rows)
    full.to_csv(f'{out}/E01_{split}_decoders_full.csv', index=False)
    main_cols = ['decoder', 'macroF1_6', 'balAcc_6', 'acc_6', 'macroF1_9', 'acc_9', 'orig_R', 'orig_FPR',
                 'ml_FS_R', 'ml_FA_R', 'ml_TS_R', 'ml_TA_R', 'ml_FS_F1', 'ml_FA_F1', 'ml_TS_F1', 'ml_TA_F1', 'ml_exact_match']
    full[main_cols].to_csv(f'{out}/E01_{split}_decoders_main.csv', index=False)
    per6 = pd.DataFrame([dict(decoder=r.decoder, cls=c, P=r[f'6_{c}_P'], R=r[f'6_{c}_R'], F1=r[f'6_{c}_F1'])
                         for _, r in full.iterrows() for c in CLASSES6])
    per9 = pd.DataFrame([dict(decoder=r.decoder, cls=c, P=r[f'9_{c}_P'], R=r[f'9_{c}_R'], F1=r[f'9_{c}_F1'])
                         for _, r in full.iterrows() for c in SHORT9])
    per6.to_csv(f'{out}/E01_{split}_per_class6.csv', index=False)
    per9.to_csv(f'{out}/E01_{split}_per_class9.csv', index=False)
    boot = paired_bootstrap(y9, preds, PAIRS)
    boot.to_csv(f'{out}/E01_{split}_paired_bootstrap.csv', index=False)
    sw = bias_sweep(cfg, df)
    sw.to_csv(f'{out}/E01_{split}_orig_bias_sweep.csv', index=False)
    sweep_plot(sw, f'{split}: per-type Recall vs Original Recall (B: b_orig sweep; A: t_b sweep)', f'{out}/E01_{split}_orig_bias_sweep.png')
    return dict(binary=bin_tab, main=full[main_cols], heads=heads, per6=per6, per9=per9, boot=boot)


# ---------------------------------------------------------------- E02
def axes_of(y9):
    a = np.array([C9_AXES[c] for c in range(9)])
    return a[y9, 0], a[y9, 1]


def error_layer(y9, p9):
    """0=正确, 1=真伪错误, 2=模态错误, 3=子类型错误（九类粒度）。"""
    ti, tt = axes_of(y9); pi, pt = axes_of(p9)
    lay = np.zeros(len(y9), int)
    wrong = y9 != p9
    auth = (y9 == 0) != (p9 == 0)
    mod = ((ti > 0) != (pi > 0)) | ((tt > 0) != (pt > 0))
    lay[wrong & auth] = 1
    lay[wrong & ~auth & mod] = 2
    lay[wrong & ~auth & ~mod] = 3
    return lay


LAYER_NAMES = {1: 'a.真伪错误', 2: 'b.模态错误', 3: 'c.子类型错误'}


def e02(df, split, dec, p9, out):
    y9 = df.y9.values
    y6, p6 = C9_TO_C6[y9], C9_TO_C6[p9]
    tag = f'{split}_{dec}'
    res = {}
    for lev, y, p, labs in [('6', y6, p6, CLASSES6), ('9', y9, p9, SHORT9)]:
        cm = confusion(y, p, len(labs))
        pd.DataFrame(cm, index=labs, columns=labs).to_csv(f'{out}/E02_{tag}_cm{lev}_count.csv')
        pd.DataFrame(cm / cm.sum(1, keepdims=True), index=labs, columns=labs).to_csv(f'{out}/E02_{tag}_cm{lev}_rownorm.csv')
        heatmap(cm, labs, f'{split} {dec} {lev}x{lev}', f'{out}/E02_{tag}_cm{lev}.png')
        res[f'cm{lev}'] = cm
    ti, tt = axes_of(y9); pi, pt = axes_of(p9)
    ai, at = (ti == pi).mean(), (tt == pt).mean()
    both = ((ti == pi) & (tt == pt)).mean()
    res['axes'] = dict(img_axis_acc=ai, txt_axis_acc=at, both_acc=both, acc_9=(y9 == p9).mean(), both_over_product=both / (ai * at))
    lay = error_layer(y9, p9)
    n_err = int((lay > 0).sum())
    rows = []
    for L in [1, 2, 3]:
        m = lay == L
        tr = pd.Series([f'{SHORT9[a]}→{SHORT9[b]}' for a, b in zip(y9[m], p9[m])]).value_counts()
        top = '; '.join(f'{k} n={v}' for k, v in tr.head(3).items())
        # 六类粒度下仍算错的部分（Mixed 内部子类型互换在六类下不算错）
        m6 = m & (y6 != p6)
        rows.append(dict(layer=LAYER_NAMES[L], n=int(m.sum()), share_of_errors9=m.sum() / n_err if n_err else np.nan,
                         n_also_wrong_6class=int(m6.sum()), top3=top))
    res['layers'] = pd.DataFrame(rows)
    res['layers'].to_csv(f'{out}/E02_{tag}_error_layers.csv', index=False)
    strat = []
    groups = [('真实九类', df.fake_cls.map(dict(zip(CLASSES9, SHORT9))), SHORT9),
              ('来源', df.source, SOURCES + ['unknown']),
              ('最大人脸尺寸', df.face_bin, SIZE_LABELS),
              ('p_fake分箱', df.pfake_bin, PFAKE_LABELS)]
    for gname, col, order in groups:
        for v in order:
            m = (col == v).values
            n = int(m.sum())
            if n == 0:
                continue
            strat.append(dict(group=gname, value=v, n=n, acc_6=(y6[m] == p6[m]).mean(), acc_9=(y9[m] == p9[m]).mean(),
                              rate_auth=(lay[m] == 1).mean(), rate_modality=(lay[m] == 2).mean(), rate_subtype=(lay[m] == 3).mean(),
                              note='样本少,仅供参考' if n < SMALL_N else ''))
    res['strat'] = pd.DataFrame(strat)
    res['strat'].to_csv(f'{out}/E02_{tag}_stratified.csv', index=False)
    # 三个描述性问题
    cm9 = res['cm9']
    n_orig = cm9[0].sum()
    i = {c: k for k, c in enumerate(SHORT9)}
    mixed_idx = [5, 6, 7, 8]
    n_mixed = cm9[mixed_idx].sum()
    q = dict(
        n_orig=int(n_orig),
        orig_to_FS=int(cm9[0, i['FS']]), orig_to_FA=int(cm9[0, i['FA']]),
        orig_to_FS_or_FA_rate=(cm9[0, i['FS']] + cm9[0, i['FA']]) / n_orig,
        orig_to_any_fake=int(n_orig - cm9[0, 0]),
        orig_to_any_image_tamper=int(cm9[0, [1, 2, 5, 6, 7, 8]].sum()),
        n_TS=int(cm9[i['TS']].sum()), TS_to_orig=int(cm9[i['TS'], 0]), TS_to_orig_rate=cm9[i['TS'], 0] / cm9[i['TS']].sum(),
        TS_to_TA=int(cm9[i['TS'], i['TA']]), TS_to_TA_rate=cm9[i['TS'], i['TA']] / cm9[i['TS']].sum(),
        n_TA=int(cm9[i['TA']].sum()), TA_to_TS=int(cm9[i['TA'], i['TS']]), TA_to_TS_rate=cm9[i['TA'], i['TS']] / cm9[i['TA']].sum(),
        TA_to_orig=int(cm9[i['TA'], 0]),
        n_mixed=int(n_mixed),
        mixed_to_image_only=int(cm9[np.ix_(mixed_idx, [1, 2])].sum()),
        mixed_to_text_only=int(cm9[np.ix_(mixed_idx, [3, 4])].sum()),
        mixed_to_single_rate=cm9[np.ix_(mixed_idx, [1, 2, 3, 4])].sum() / n_mixed,
        mixed_to_orig=int(cm9[mixed_idx, 0].sum()),
    )
    res['questions'] = q
    json.dump(dict(axes=res['axes'], questions=q), open(f'{out}/E02_{tag}_axes_questions.json', 'w'), indent=2, default=float)
    return res


# ---------------------------------------------------------------- E03
def ece(p, t, bins=15):
    e = np.linspace(0, 1, bins + 1)
    b = np.clip(np.digitize(p, e[1:-1]), 0, bins - 1)
    tot = 0.
    for k in range(bins):
        m = b == k
        if m.any():
            tot += m.mean() * abs(p[m].mean() - t[m].mean())
    return tot


def e03(cfg, df, split, preds, out):
    pf, P = probs(df)
    y9 = df.y9.values
    y6 = C9_TO_C6[y9]
    yb = y9 != 0
    bf = pf >= 0.5
    conf = {
        'bin假_四维全<0.5': bf & (P.max(1) < 0.5),
        'bin真_任一维>=0.5': ~bf & (P.max(1) >= 0.5),
        'bin真_任一维>=0.9': ~bf & (P.max(1) >= 0.9),
    }
    rows = []
    for ci, c in enumerate(CLASSES6 + ['全部']):
        m = (y6 == ci) if c != '全部' else np.ones(len(y6), bool)
        r = dict(cls6=c, n=int(m.sum()))
        for k, v in conf.items():
            r[f'{k}_n'] = int((v & m).sum()); r[f'{k}_rate'] = (v & m).sum() / m.sum()
        r['note'] = '样本少,仅供参考' if m.sum() < SMALL_N else ''
        rows.append(r)
    rates = pd.DataFrame(rows)
    rates.to_csv(f'{out}/E03_{split}_conflict_rates.csv', index=False)
    err = {'bin@0.5': bf != yb, **{f'{d}_6cls': C9_TO_C6[preds[d]] != y6 for d in E02_DECODERS},
           **{f'{d}_9cls': preds[d] != y9 for d in E02_DECODERS}}
    rows = []
    for k, v in conf.items():
        for side, m in [('冲突', v), ('非冲突', ~v)]:
            r = dict(conflict=k, side=side, n=int(m.sum()))
            for en, e in err.items():
                r[f'err_{en}'] = e[m].mean() if m.any() else np.nan
            r['note'] = '样本少,仅供参考' if m.sum() < SMALL_N else ''
            rows.append(r)
    errs = pd.DataFrame(rows)
    errs.to_csv(f'{out}/E03_{split}_conflict_error_rates.csv', index=False)
    T = df[[f't_{d}' for d in DIMS]].values
    Pt = apply_temps(P, cfg['B_TS_b0']['temps'])
    cal = pd.DataFrame([dict(dim=d, T=cfg['B_TS_b0']['temps'][k], ece_raw=ece(P[:, k], T[:, k]), ece_ts=ece(Pt[:, k], T[:, k]))
                        for k, d in enumerate(DIMS)])
    cal.to_csv(f'{out}/E03_{split}_calibration_ece.csv', index=False)
    return dict(rates=rates, errs=errs, cal=cal)


def report_split(cfg, df, split, do_e02=True):
    out = {k: os.path.join(RES_DIR, k) for k in ['E01', 'E02', 'E03']}
    for v in out.values():
        os.makedirs(v, exist_ok=True)
    preds = predict_all(cfg, df)
    pd.DataFrame({'meta_index': df.meta_index.values, 'y9': df.y9.values, **preds}).to_csv(
        f"{out['E01']}/E01_{split}_preds9.csv.gz", index=False, compression='gzip')
    res = dict(e01=e01(cfg, df, split, preds, out['E01']), e03=e03(cfg, df, split, preds, out['E03']))
    if do_e02:
        res['e02'] = {d: e02(df, split, d, preds[d], out['E02']) for d in E02_DECODERS}
    return res
