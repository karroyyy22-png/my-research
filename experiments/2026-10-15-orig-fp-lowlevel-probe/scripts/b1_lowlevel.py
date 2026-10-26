"""B1：图像轴——Original 误报与低层特征（只用 CPU，读图，最多 4 进程）。
用法: nice -n 19 python b1_lowlevel.py --split val|test [--n 500] [--workers 4]
规则全部来自 frozen_hypotheses.json 的 B1 / twins / source_field 段；本脚本不做任何判定以外的选择。
输出 results/B1/<split>/（--n 时为 results/smoke/B1/<split>/）：
  features_per_image.csv  逐图特征（不含图片）    per_sample.csv  逐样本 meta_index+分组+特征
  missingness.csv  range.csv  matching.csv  matched_pairs.csv
  a_matched.csv  b_logit_single.csv  b_logit_joint.csv  spearman.csv
  c_dataset_bias.csv  d_fsfa_miss.csv  e_wh_source.csv  twins_pairs.csv  twins_summary.csv  split_status.csv
"""
import os, sys, math, json, argparse
from multiprocessing import Pool
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from b_common import check_frozen, load_split, boot_diff_props, out_dir  # noqa: E402
from b1_features import image_features, QUALITY, FACE_FEATS, ENCODING  # noqa: E402
from probe_lowlevel import auc  # noqa: E402

BUCKET_EDGES = [0, 32, 64, 128, 256, np.inf]
BUCKET_LABELS = ['<32', '32-64', '64-128', '128-256', '>=256']
LOG1P = ['lap_var', 'face_lap_var', 'noise_sigma', 'noise_sigma_flat', 'face_noise_sigma', 'bytes_per_px']
GEOM = ['W', 'H', 'max_face', 'n_faces']
# frozen v3 predictions_v3.directed：+1 = FP 更高/更多，-1 = FP 更低
DIRECTED = {'noise_sigma': 1, 'noise_sigma_flat': 1, 'face_noise_sigma': 1,
            'lap_var': -1, 'face_lap_var': -1, 'mean_saturation': -1, 'near_gray': 1}
JPEG_NOTE = 'jpeg_q 在数据中是常数（同一张量化表），H1 的“压缩”一项不可检验（保留在假设中，标注不可检验）'


# ---------------------------------------------------------------- 工具
def face_geom(mt):
    if len(mt) == 0:
        return 0.0, 0
    return float(np.sqrt(((mt[:, 2] - mt[:, 0]) * (mt[:, 3] - mt[:, 1])).max())), len(mt)


def bucket(mf, n):
    if n == 0:
        return 'none'
    return BUCKET_LABELS[int(np.searchsorted(BUCKET_EDGES, mf, side='right') - 1)]


def smd(x1, x0):
    s = math.sqrt((np.var(x1, ddof=1) + np.var(x0, ddof=1)) / 2) if len(x1) > 1 and len(x0) > 1 else np.nan
    return (np.mean(x1) - np.mean(x0)) / s if s and s > 0 else np.nan


def fast_auc(s1, s0):
    """Mann-Whitney AUC（s1 为正类）；与 probe_lowlevel.auc 等价，用于 bootstrap 提速。"""
    from scipy.stats import rankdata
    r = rankdata(np.concatenate([s1, s0]))
    n1, n0 = len(s1), len(s0)
    return (r[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0) if n1 and n0 else np.nan


def holm(p):
    p = np.asarray(p, float); out = np.full_like(p, np.nan)
    ok = ~np.isnan(p); idx = np.where(ok)[0]
    if len(idx) == 0:
        return out
    order = idx[np.argsort(p[idx])]; m = len(order); run = 0.0
    for k, i in enumerate(order):
        run = max(run, min(1.0, (m - k) * p[i])); out[i] = run
    return out


def logit_irls(X, y, iters=100):
    """无惩罚 logistic 回归（Newton/IRLS），返回 (beta, se)；不收敛或奇异返回 nan。"""
    beta = np.zeros(X.shape[1])
    for _ in range(iters):
        eta = np.clip(X @ beta, -30, 30); p = 1 / (1 + np.exp(-eta))
        W = p * (1 - p); H = X.T @ (X * W[:, None]); g = X.T @ (y - p)
        try:
            step = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            return np.full(X.shape[1], np.nan), np.full(X.shape[1], np.nan)
        beta += step
        if np.max(np.abs(step)) < 1e-8:
            break
    try:
        se = np.sqrt(np.diag(np.linalg.inv(H)))
    except np.linalg.LinAlgError:
        se = np.full(X.shape[1], np.nan)
    return beta, se


def vif(Z):
    out = []
    for j in range(Z.shape[1]):
        A = np.column_stack([np.ones(len(Z)), np.delete(Z, j, axis=1)])
        coef, *_ = np.linalg.lstsq(A, Z[:, j], rcond=None)
        res = Z[:, j] - A @ coef; r2 = 1 - res.var() / Z[:, j].var() if Z[:, j].var() > 0 else np.nan
        out.append(1 / (1 - r2) if r2 < 1 else np.inf)
    return out


def match_pairs(fp, tn, seed):
    """fp/tn: DataFrame，含 key 列。每个 TN 最多用一次；FP 随机顺序；候选中均匀随机。返回 [(fp_idx, tn_idx)]。"""
    rng = np.random.default_rng(seed)
    pool = {}
    for i, k in zip(tn.index, tn.key):
        pool.setdefault(k, []).append(i)
    pairs = []
    for i in fp.index[rng.permutation(len(fp))]:
        cand = pool.get(fp.at[i, 'key'])
        if cand:
            pairs.append((i, cand.pop(int(rng.integers(len(cand))))))
    return pairs


def jaccard_words(a, b):
    a, b = set(a.lower().split()), set(b.lower().split())
    return len(a & b) / max(1, len(a | b))


# ---------------------------------------------------------------- 主流程
def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', choices=['val', 'test'], default='val')
    ap.add_argument('--n', type=int, default=0, help='小样本：随机抽 n 条 Original，其余组按同比例抽样')
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--tag', default='', help='输出子目录后缀，例如 _v5 → results/B1/val_v5/')
    a = ap.parse_args(argv)
    cfg, sha = check_frozen(a.split)
    assert a.workers <= cfg['run']['max_workers']
    seed, B = cfg['seed'], cfg['bootstrap']['B']
    od = out_dir('B1', a.split + a.tag, a.n > 0)
    rng = np.random.default_rng(seed)

    df, _ = load_split(cfg, a.split)
    geo = [face_geom(m) for m in df.mtcnn]
    df['max_face'] = [g[0] for g in geo]; df['n_faces'] = [g[1] for g in geo]
    df['bucket'] = [bucket(mf, n) for mf, n in geo]
    df['key'] = list(zip(df.source, df.bucket, df.W, df.H))
    df['group'] = df.fake_cls.map({'orig': 'orig', 'face_swap': 'FS', 'face_attribute': 'FA',
                                   'text_swap': 'TS', 'text_attribute': 'TA'}).fillna('mixed')

    # 全量协变量匹配率（不读图，只用来源/脸桶/W/H 与 FP 标签）
    o_all = df[df.group == 'orig'].copy(); o_all['fp'] = (o_all.prob_fake >= 0.5).astype(int)
    full_pairs = match_pairs(o_all[o_all.fp == 1], o_all[o_all.fp == 0], seed)
    full_match = dict(scope='full_split_covariates_only', n_fp=int(o_all.fp.sum()), n_tn=int((o_all.fp == 0).sum()),
                      n_matched=len(full_pairs), match_rate=len(full_pairs) / max(1, o_all.fp.sum()))
    fm = {i for i, _ in full_pairs}
    full_by_src = [dict(scope=f'full_split_covariates_only:{s}', n_fp=len(g), n_matched=int(g.index.isin(fm).sum()),
                        n_dropped=int((~g.index.isin(fm)).sum()), match_rate=g.index.isin(fm).mean())
                   for s, g in o_all[o_all.fp == 1].groupby('source')]
    low_sources = [r['scope'].split(':', 1)[1] for r in full_by_src if r['match_rate'] < 0.70]  # 次要分析来源（v3）

    # 抽样
    if a.n:
        frac = min(1.0, a.n / len(o_all))
        o = o_all.sample(n=min(a.n, len(o_all)), random_state=seed)
        others = df[df.group != 'orig'].groupby('group', group_keys=False).apply(
            lambda g: g.sample(frac=frac, random_state=seed))
        df = pd.concat([o.drop(columns='fp'), others])
    o = df[df.group == 'orig'].copy(); o['fp'] = (o.prob_fake >= 0.5).astype(int)

    # 读图（按 image 去重；orig/TS/TA 共用原图）
    need = df.drop_duplicates('image')
    jobs = list(zip(need.image, need.mtcnn))
    print(f'[img] {len(jobs)} 张唯一图像, workers={a.workers}')
    with Pool(a.workers) as p:
        feats = pd.DataFrame(p.map(image_features, jobs, chunksize=32))
    feats.to_csv(f'{od}/features_per_image.csv', index=False)
    ok = feats[feats.err == ''].drop(columns='err')
    df = df.merge(ok, on='image', how='left')
    df['read_fail'] = df.lap_var.isna().astype(int)
    df['face_8_16'] = ((df.max_face >= 8) & (df.max_face < 16)).astype(int)
    keep = ['meta_index', 'group', 'source', 'W', 'H', 'max_face', 'n_faces', 'bucket', 'prob_fake', 'read_fail',
            'face_small', 'face_8_16', 'jpeg_q_invalid', 'bytes_per_px_invalid', *QUALITY]
    df[keep].to_csv(f'{od}/per_sample.csv', index=False)
    o = df[df.group == 'orig'].copy(); o['fp'] = (o.prob_fake >= 0.5).astype(int)

    # ---- 缺失
    miss_items = {
        'read_fail': o.read_fail, 'no_mtcnn': (o.n_faces == 0).astype(int),
        'face_small_lt8': o.face_small.fillna(0).astype(int), 'face_8_16': o.face_8_16,
        'jpeg_q_invalid_any': (o.jpeg_q_invalid.fillna('') != '').astype(int),
        **{f'jpeg_q_invalid_{r}': (o.jpeg_q_invalid == r).astype(int) for r in ['non_jpeg', 'q_ge_98', 'non_ijg']},
        'bytes_per_px_invalid': o.bytes_per_px_invalid.fillna(0).astype(int),
        **{f'nan_{q}': o[q].isna().astype(int) for q in QUALITY}}
    rows = []
    for k, v in miss_items.items():
        v1, v0 = v[o.fp == 1], v[o.fp == 0]
        lo, hi = boot_diff_props(v1, v0, B, rng)
        d = v1.mean() - v0.mean()
        rows.append(dict(item=k, n_fp=len(v1), count_fp=int(v1.sum()), rate_fp=v1.mean(), n_tn=len(v0),
                         count_tn=int(v0.sum()), rate_tn=v0.mean(), diff=d, ci_lo=lo, ci_hi=hi,
                         flag_marked_diff=bool(abs(d) >= 0.02 and (lo > 0 or hi < 0))))
    miss = pd.DataFrame(rows); miss.to_csv(f'{od}/missingness.csv', index=False)

    # ---- 取值范围
    rows = []
    for q in QUALITY + GEOM:
        for grp, g in (('FP', o[o.fp == 1]), ('TN', o[o.fp == 0])):
            x = g[q].dropna()
            rows.append(dict(feature=q, group=grp, n=len(g), n_valid=len(x), missing_rate=1 - len(x) / max(1, len(g)),
                             min=x.min(), p01=x.quantile(.01), median=x.median(), p99=x.quantile(.99), max=x.max()))
    pd.DataFrame(rows).to_csv(f'{od}/range.csv', index=False)

    # ---- (a) 精确匹配
    pairs = match_pairs(o[o.fp == 1], o[o.fp == 0], seed)
    fp_all = o[o.fp == 1]
    matched_fp = {i for i, _ in pairs}
    unm = fp_all[~fp_all.index.isin(matched_fp)]
    mrows = [full_match, *full_by_src, dict(scope='this_run', n_fp=len(fp_all), n_tn=int((o.fp == 0).sum()), n_matched=len(pairs),
                              match_rate=len(pairs) / max(1, len(fp_all)))]
    for s, g in fp_all.groupby('source'):
        n_un = int(g.index.isin(unm.index).sum())
        mrows.append(dict(scope=f'this_run:{s}', n_fp=len(g), n_matched=len(g) - n_un, n_dropped=n_un,
                          match_rate=(len(g) - n_un) / len(g)))
    pd.DataFrame(mrows).to_csv(f'{od}/matching.csv', index=False)
    pf, pt = o.loc[[i for i, _ in pairs]].reset_index(drop=True), o.loc[[j for _, j in pairs]].reset_index(drop=True)
    mp = pd.concat([pf.add_prefix('fp_'), pt.add_prefix('tn_')], axis=1)
    mp[[f'{s}_{c}' for c in ['meta_index', 'source', 'W', 'H', 'bucket', 'prob_fake', *QUALITY] for s in ('fp', 'tn')]
       ].to_csv(f'{od}/matched_pairs.csv', index=False)

    def matched_stats(feats, sub_mask=None, tag='all', pf=pf, pt=pt):
        rows = []
        for q in feats:
            ok_ = pf[q].notna() & pt[q].notna()
            if sub_mask is not None:
                ok_ &= sub_mask
            x1, x0 = pf.loc[ok_, q].values, pt.loc[ok_, q].values
            n = len(x1)
            r = dict(feature=q, subset=tag, n_pairs=n, mean_fp=np.mean(x1) if n else np.nan,
                     mean_tn=np.mean(x0) if n else np.nan, smd=smd(x1, x0) if n > 1 else np.nan,
                     auc=fast_auc(x1, x0) if n else np.nan,
                     p_mwu=mannwhitneyu(x1, x0, alternative='two-sided').pvalue if n > 1 and np.ptp(np.r_[x1, x0]) > 0 else np.nan)
            if n > 1:
                bs_auc, bs_smd = [], []
                for _ in range(B):
                    k = rng.integers(0, n, n)
                    bs_auc.append(fast_auc(x1[k], x0[k])); bs_smd.append(smd(x1[k], x0[k]))
                r.update(auc_ci_lo=np.nanpercentile(bs_auc, 2.5), auc_ci_hi=np.nanpercentile(bs_auc, 97.5),
                         smd_ci_lo=np.nanpercentile(bs_smd, 2.5), smd_ci_hi=np.nanpercentile(bs_smd, 97.5))
            rows.append(r)
        return pd.DataFrame(rows)
    # v2：常数特征不可检验；脸部特征主分析只用两侧最大脸 >=16 的配对，全部配对另列为参考
    testable = [q for q in QUALITY if o[q].dropna().nunique() >= 2]
    constant = [q for q in QUALITY if q not in testable]
    face16 = (pf.max_face >= 16) & (pt.max_face >= 16)
    A = pd.concat([matched_stats([q for q in testable if q not in FACE_FEATS], tag='primary'),
                   matched_stats([q for q in testable if q in FACE_FEATS], face16, tag='primary(max_face>=16)')],
                  ignore_index=True)
    A['p_holm'] = holm(A.p_mwu.values)
    Aref = matched_stats([q for q in testable if q in FACE_FEATS], tag='reference(all faces)')
    Aref['p_holm'] = np.nan
    Acon = pd.DataFrame([dict(feature=q, subset='constant_untestable') for q in constant])
    A['category'] = A.feature.map(lambda q: 'encoding_trace' if q in ENCODING else 'face' if q in FACE_FEATS else 'imaging/color')
    pd.concat([A, Aref, Acon], ignore_index=True).to_csv(f'{od}/a_matched.csv', index=False)

    # ---- v3 次要分析：全量协变量匹配率 <70% 的来源，仅按 (来源, W, H) 匹配（不用于判定）
    sec_rows, sec_match = [], []
    if low_sources:
        o2 = o[o.source.isin(low_sources)].copy(); o2['key'] = list(zip(o2.source, o2.W, o2.H))
        p2 = match_pairs(o2[o2.fp == 1], o2[o2.fp == 0], seed)
        for s in low_sources + ['pooled']:
            sel = [(i, j) for i, j in p2 if s == 'pooled' or o2.at[i, 'source'] == s]
            nfp = int(((o2.fp == 1) & ((o2.source == s) | (s == 'pooled'))).sum())
            sec_match.append(dict(scope=f'secondary(source,W,H):{s}', n_fp=nfp, n_matched=len(sel),
                                  n_dropped=nfp - len(sel), match_rate=len(sel) / max(1, nfp)))
            if len(sel) > 1:
                pf2 = o2.loc[[i for i, _ in sel]].reset_index(drop=True); pt2 = o2.loc[[j for _, j in sel]].reset_index(drop=True)
                f16 = (pf2.max_face >= 16) & (pt2.max_face >= 16)
                t2 = pd.concat([matched_stats([q for q in testable if q not in FACE_FEATS], tag=f'secondary:{s}', pf=pf2, pt=pt2),
                                matched_stats([q for q in testable if q in FACE_FEATS], f16, tag=f'secondary:{s}(max_face>=16)',
                                              pf=pf2, pt=pt2)], ignore_index=True)
                sec_rows.append(t2)
    pd.DataFrame(sec_match, columns=['scope', 'n_fp', 'n_matched', 'n_dropped', 'match_rate']).to_csv(
        f'{od}/a_secondary_matching.csv', index=False)
    (pd.concat(sec_rows, ignore_index=True) if sec_rows else pd.DataFrame(columns=['feature', 'subset'])).assign(
        low_sources=','.join(low_sources)).to_csv(f'{od}/a_secondary_matched.csv', index=False)

    # ---- (b) logistic 回归（全体 orig，完全样本）
    src_levels = [s for s in sorted(o.source.unique()) if s != 'guardian']
    cov = pd.DataFrame({f'src_{s}': (o.source == s).astype(float) for s in src_levels}, index=o.index)
    cov['log_WH'] = np.log(o.W * o.H); cov['log1p_maxface'] = np.log1p(o.max_face)
    cov['has_face'] = (o.n_faces > 0).astype(float); cov['n_faces'] = o.n_faces.astype(float)
    Zq = pd.DataFrame(index=o.index)
    for q in QUALITY:
        x = np.log1p(o[q]) if q in LOG1P else o[q].astype(float)
        sd = x.std()
        Zq[q] = (x - x.mean()) / sd if sd and sd > 0 else np.nan
    y = o.fp.values.astype(float)
    rows = []
    for q in testable:
        okm = Zq[q].notna().values
        if q in FACE_FEATS:
            okm &= (o.max_face >= 16).values
        if okm.sum() < 20:
            rows.append(dict(feature=q, n=int(okm.sum()))); continue
        cv = cov[okm]; cv = cv.loc[:, cv.std() > 0]
        X = np.column_stack([np.ones(okm.sum()), Zq.loc[okm, q].values, cv.values])
        b, se = logit_irls(X, y[okm])
        z = b[1] / se[1] if se[1] > 0 else np.nan
        from scipy.stats import norm
        rows.append(dict(feature=q, n=int(okm.sum()), coef=b[1], se=se[1], ci_lo=b[1] - 1.96 * se[1],
                         ci_hi=b[1] + 1.96 * se[1], p=2 * norm.sf(abs(z)) if z == z else np.nan))
    Bs = pd.DataFrame(rows); Bs['p_holm'] = holm(Bs.p.values) if 'p' in Bs else np.nan
    Bs.to_csv(f'{od}/b_logit_single.csv', index=False)
    okj = Zq[testable].notna().all(axis=1).values & (o.max_face >= 16).values
    joint_rows = []
    qcols = [q for q in testable if Zq.loc[okj, q].std() > 0] if okj.sum() else []
    if okj.sum() > 50 and qcols:
        cv = cov[okj]; cv = cv.loc[:, cv.std() > 0]
        Zj = np.column_stack([Zq.loc[okj, qcols].values, cv.values])
        b, se = logit_irls(np.column_stack([np.ones(okj.sum()), Zj]), y[okj])
        vifs = vif(Zj)
        for k, q in enumerate(qcols):
            joint_rows.append(dict(feature=q, n=int(okj.sum()), coef=b[k + 1], se=se[k + 1],
                                   ci_lo=b[k + 1] - 1.96 * se[k + 1], ci_hi=b[k + 1] + 1.96 * se[k + 1], vif=vifs[k]))
    pd.DataFrame(joint_rows).to_csv(f'{od}/b_logit_joint.csv', index=False)

    # ---- Spearman
    o[QUALITY].corr(method='spearman').to_csv(f'{od}/spearman.csv')

    # ---- (c) 数据集偏置：图像未改（orig/TS/TA 唯一图像）vs FS / FA
    un = df[df.group.isin(['orig', 'TS', 'TA'])].drop_duplicates('image')
    rows = []
    for c in ['FS', 'FA']:
        fk = df[df.group == c]
        for q in QUALITY:
            x1, x0 = fk[q].dropna().values, un[q].dropna().values
            const = np.unique(np.r_[x1, x0]).size < 2  # 常数特征（如 jpeg_q）输出空值
            rows.append(dict(contrast=f'{c} vs unaltered(orig/TS/TA unique images)', feature=q, n_manip=len(x1),
                             n_unaltered=len(x0), auc_manip=np.nan if const else fast_auc(x1, x0),
                             smd=np.nan if const else smd(x1, x0),
                             note='常数，不可检验' if const else
                             '生成/重编码过程的痕迹（不解读为模型捷径）' if q in ENCODING else ''))
    pd.DataFrame(rows).to_csv(f'{od}/c_dataset_bias.csv', index=False)

    # ---- (d) FS/FA 漏检 vs 检出
    rows = []
    for c in ['FS', 'FA']:
        g = df[df.group == c]; miss_ = g.prob_fake < 0.5
        for q in QUALITY:
            x1, x0 = g.loc[miss_, q].dropna().values, g.loc[~miss_, q].dropna().values
            const = np.unique(np.r_[x1, x0]).size < 2
            rows.append(dict(cls=c, feature=q, n_missed=len(x1), n_detected=len(x0),
                             auc_missed=np.nan if const else fast_auc(x1, x0), smd=np.nan if const else smd(x1, x0),
                             note='常数，不可检验' if const else ''))
    pd.DataFrame(rows).to_csv(f'{od}/d_fsfa_miss.csv', index=False)

    # ---- H2（v5，探索性假设，val 上提出）：face_bg_ratio 三个比较，以 id 为簇 bootstrap
    def id_cluster_auc(pos, neg, col):
        d = pd.concat([pos[['id', col]].assign(y=1), neg[['id', col]].assign(y=0)]).dropna(subset=[col])
        ids, inv = np.unique(d.id.values, return_inverse=True)
        v, y = d[col].values, d.y.values
        bs = []
        for _ in range(B):
            w = np.bincount(rng.integers(0, len(ids), len(ids)), minlength=len(ids))[inv]
            vv, yy = np.repeat(v, w), np.repeat(y, w)
            bs.append(fast_auc(vv[yy == 1], vv[yy == 0]))
        return dict(n_pos=int(d.y.sum()), n_neg=int((d.y == 0).sum()), n_clusters=len(ids),
                    auc=fast_auc(v[y == 1], v[y == 0]), ci_lo=np.nanpercentile(bs, 2.5), ci_hi=np.nanpercentile(bs, 97.5))
    h2 = []
    r1 = A[A.feature == 'face_bg_ratio'].iloc[0]  # 匹配对（最大脸>=16），每对两条样本 id 不同，按对重采样即按 id 簇
    h2.append(dict(comparison='H2-1 Original FP vs 配对 TN', pred='AUC<0.5（FP 更低）', n_pos=int(r1.n_pairs),
                   n_neg=int(r1.n_pairs), n_clusters=int(r1.n_pairs), auc=r1.auc, ci_lo=r1.auc_ci_lo, ci_hi=r1.auc_ci_hi))
    fsfa = df[df.group.isin(['FS', 'FA'])]
    h2.append(dict(comparison='H2-2 FS+FA vs 未改图（orig/TS/TA 唯一图像）', pred='AUC<0.5（篡改图更低）',
                   **id_cluster_auc(fsfa, un, 'face_bg_ratio')))
    fa = df[df.group == 'FA']
    h2.append(dict(comparison='H2-3 漏检 FA vs 检出 FA', pred='AUC>0.5（漏检更高）',
                   **id_cluster_auc(fa[fa.prob_fake < 0.5], fa[fa.prob_fake >= 0.5], 'face_bg_ratio')))
    for c in ['FS', 'FA']:  # 仅描述
        h2.append(dict(comparison=f'(描述) {c} vs 未改图', pred='',
                       **id_cluster_auc(df[df.group == c], un, 'face_bg_ratio')))
    h2 = pd.DataFrame(h2)
    want = {'H2-1': -1, 'H2-2': -1, 'H2-3': 1}
    h2['pred_dir'] = h2.comparison.str[:4].map(want)
    h2['dir_as_predicted'] = (np.sign(h2.auc - 0.5) == h2.pred_dir).where(h2.pred_dir.notna())  # 描述行无预测方向，留空
    h2['ci_excl_05'] = (h2.ci_lo > 0.5) | (h2.ci_hi < 0.5)
    h2['note'] = '探索性假设（val 上提出）；即使确认，效应小，不能解释大部分误报' + \
        ('；val 为探索集上的描述，不是确认' if a.split == 'val' else '')
    h2['split'] = a.split; h2['frozen_sha256'] = sha
    h2.to_csv(f'{od}/h2_face_bg_ratio.csv', index=False)

    # H2 调整版（v6 adjusted_supplement_v6）：补充，不参与确认判定
    def adj_coef(pos, neg):
        d = pd.concat([pos.assign(y=1), neg.assign(y=0)]).dropna(subset=['face_bg_ratio'])
        x = d.face_bg_ratio.values; x = (x - x.mean()) / x.std()
        X = np.column_stack([np.ones(len(d)), x, np.log1p(d.max_face.values), np.log(d.W.values * d.H.values)])
        y_ = d.y.values.astype(float)
        b, se = logit_irls(X, y_)
        ids, inv = np.unique(d.id.values, return_inverse=True)
        bs = []
        for _ in range(B):
            w = np.bincount(rng.integers(0, len(ids), len(ids)), minlength=len(ids))[inv]
            k = np.repeat(np.arange(len(d)), w)
            bs.append(logit_irls(X[k], y_[k])[0][1])
        return dict(n_pos=int(y_.sum()), n_neg=int((1 - y_).sum()), coef_adj=b[1],
                    ci_lo=np.nanpercentile(bs, 2.5), ci_hi=np.nanpercentile(bs, 97.5))
    adj = [dict(comparison='H2-2 FS+FA vs 未改图', pred_dir=-1, **adj_coef(fsfa, un)),
           dict(comparison='H2-3 漏检 FA vs 检出 FA', pred_dir=1, **adj_coef(fa[fa.prob_fake < 0.5], fa[fa.prob_fake >= 0.5]))]
    adj = pd.DataFrame(adj)
    unadj = h2.set_index(h2.comparison.str[:4]).auc
    adj['auc_unadjusted'] = [unadj['H2-2'], unadj['H2-3']]
    adj['direction_agrees_with_unadjusted'] = np.sign(adj.coef_adj) == np.sign(adj.auc_unadjusted - 0.5)
    adj['label'] = '补充，不参与确认判定（控制 log1p(max_face)、log(W*H)）'
    adj['split'] = a.split; adj['frozen_sha256'] = sha
    adj.to_csv(f'{od}/h2_adjusted_supplement.csv', index=False)

    # ---- (e) (W,H)→来源纯度
    si = cfg['source_field']
    t = df[df.group.isin(['orig', 'TS', 'TA'])].drop_duplicates('image')
    wh = t.groupby(['W', 'H']).source.agg(n='size', top=lambda s: s.value_counts().iloc[0] / len(s),
                                          mode=lambda s: s.value_counts().index[0]).reset_index()
    wh['pure'] = (wh.n >= 20) & (wh.top >= 0.95)
    cover_un = wh.loc[wh.pure, 'n'].sum() / max(1, wh.n.sum())
    infer = cover_un >= 0.5
    pure_map = {(r.W, r.H): r['mode'] for _, r in wh[wh.pure].iterrows()}
    fsm = df[df.group.isin(['FS', 'FA', 'mixed'])]
    cov_m = np.mean([(w, h) in pure_map for w, h in zip(fsm.W, fsm.H)]) if infer and len(fsm) else 0.0
    wh.to_csv(f'{od}/e_wh_source_keys.csv', index=False)
    pd.DataFrame([dict(n_unaltered_images=int(wh.n.sum()), n_keys=len(wh), n_pure_keys=int(wh.pure.sum()),
                       coverage_unaltered=cover_un, inference_enabled=infer,
                       coverage_fs_fa_mixed=cov_m, rule=si['inference'])]).to_csv(f'{od}/e_wh_source.csv', index=False)

    # ---- twins（同目录、caption Jaccard>=0.8、(W,H) 与脸桶相同；TN 不放回；以目录为簇 bootstrap）
    tw = []
    for folder, g in o.groupby('folder'):
        fpg, tng = g[g.fp == 1], g[g.fp == 0]
        if len(fpg) == 0 or len(tng) == 0:
            continue
        used = set()
        for i in fpg.index[rng.permutation(len(fpg))]:
            f = o.loc[i]
            cands = [(jaccard_words(f.text, o.at[j, 'text']), j) for j in tng.index
                     if j not in used and o.at[j, 'W'] == f.W and o.at[j, 'H'] == f.H and o.at[j, 'bucket'] == f.bucket]
            cands = [c for c in cands if c[0] >= 0.8]
            if not cands:
                continue
            best = max(c[0] for c in cands)
            tie = [j for jj, j in cands if jj == best]
            j = tie[int(rng.integers(len(tie)))]; used.add(j)
            tw.append(dict(folder=folder, fp_meta=f.meta_index, tn_meta=o.at[j, 'meta_index'], jaccard=best,
                           **{f'd_{q}': f[q] - o.at[j, q] for q in QUALITY}))
    tw = pd.DataFrame(tw); tw.to_csv(f'{od}/twins_pairs.csv', index=False)
    rows = []
    if len(tw):
        folders = tw.folder.unique()
        for q in QUALITY:
            d = tw[['folder', f'd_{q}']].dropna()
            if len(d) == 0:
                rows.append(dict(feature=q, n_pairs=0)); continue
            byf = {f: g[f'd_{q}'].values for f, g in d.groupby('folder')}
            fl = np.array(list(byf))
            bm, bh = [], []
            for _ in range(B):
                v = np.concatenate([byf[f] for f in fl[rng.integers(0, len(fl), len(fl))]])
                bm.append(v.mean()); bh.append((v > 0).mean())
            rows.append(dict(feature=q, n_pairs=len(d), n_clusters=len(fl), mean_diff=d[f'd_{q}'].mean(),
                             mean_ci_lo=np.percentile(bm, 2.5), mean_ci_hi=np.percentile(bm, 97.5),
                             frac_fp_higher=(d[f'd_{q}'] > 0).mean(),
                             frac_ci_lo=np.percentile(bh, 2.5), frac_ci_hi=np.percentile(bh, 97.5)))
        del folders
    pd.DataFrame(rows, columns=['feature', 'n_pairs', 'n_clusters', 'mean_diff', 'mean_ci_lo', 'mean_ci_hi',
                                'frac_fp_higher', 'frac_ci_lo', 'frac_ci_hi']).to_csv(f'{od}/twins_summary.csv', index=False)

    # ---- 本 split 的判据条件（最终判定需 val 与 test 合并，见 b1_verdict.py）
    st = A.merge(Bs[['feature', 'coef', 'ci_lo', 'ci_hi', 'p', 'p_holm']].rename(
        columns={'ci_lo': 'coef_ci_lo', 'ci_hi': 'coef_ci_hi', 'p': 'p_coef', 'p_holm': 'p_coef_holm'}), on='feature', how='left')
    st['auc_effect'] = (st.auc >= 0.60) | (st.auc <= 0.40)
    st['auc_ci_excl_05'] = (st.auc_ci_lo > 0.5) | (st.auc_ci_hi < 0.5)
    st['auc_ci_in_band'] = (st.auc_ci_lo >= 0.45) & (st.auc_ci_hi <= 0.55)
    st['coef_ci_excl_0'] = (st.coef_ci_lo > 0) | (st.coef_ci_hi < 0)
    st['coef_same_dir'] = np.sign(st.coef) == np.sign(st.auc - 0.5)
    st['auc_holm_sig'] = st.p_holm < 0.05; st['coef_holm_sig'] = st.p_coef_holm < 0.05
    st['split'] = a.split; st['frozen_sha256'] = sha; st['n_features_tested'] = len(testable)
    st['constant_excluded'] = ','.join(constant)
    # v3 方向：pred_dir=+1/-1（有预测方向），0 = 无方向，只做双侧
    st['pred_dir'] = st.feature.map(DIRECTED).fillna(0).astype(int)
    dev = st.auc - 0.5
    st['auc_dev_ge_010'] = dev.abs() >= 0.10
    st['in_pred_dir'] = (st.pred_dir != 0) & (np.sign(dev) == st.pred_dir)
    st['opposite_to_H1'] = (st.pred_dir != 0) & (np.sign(dev) == -st.pred_dir) & st.auc_dev_ge_010
    st['coef_in_pred_dir'] = (st.pred_dir != 0) & (np.sign(st.coef) == st.pred_dir)
    st['jpeg_note'] = JPEG_NOTE if 'jpeg_q' in constant else ''
    st.to_csv(f'{od}/split_status.csv', index=False)

    if a.n:  # 小样本：随机 8 对的特征值
        k = min(8, len(mp))
        mp.sample(n=k, random_state=seed)[[f'{s}_{c}' for c in ['meta_index', 'source', 'W', 'H', 'bucket', 'prob_fake',
                                                                  *QUALITY] for s in ('fp', 'tn')]
                                          ].to_csv(f'{od}/sample8_pairs.csv', index=False)
    print(f'[done] {od}  frozen_sha256={sha}')


if __name__ == '__main__':
    main(sys.argv[1:])
