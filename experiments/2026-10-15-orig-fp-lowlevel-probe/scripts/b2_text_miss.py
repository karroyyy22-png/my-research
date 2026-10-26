"""B2：ASAP 文本篡改（TS/TA）漏检分析。机械实现 results/B2/frozen_hypotheses_B2.json（H-T1/H-T2/H-T3）。
只读 pkl、metadata 与 exp/6class-decoder-diag 分支上的 E01 A_tuned 预测和 val 划分（git show 读取，不切换分支）；只用 CPU。
用法: python b2_text_miss.py --split val_holdout            -> results/B2/val/
      python b2_text_miss.py --split test --confirm_test    -> results/B2/test/（val 结果经确认后只跑一次）
"""
import os, sys, io, json, hashlib, pickle, argparse, subprocess
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

EXP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = os.path.dirname(os.path.dirname(EXP_DIR))
B2_DIR = os.path.join(EXP_DIR, 'results', 'B2')
FROZEN = os.path.join(B2_DIR, 'frozen_hypotheses_B2.json')
FROZEN_SHA = os.path.join(B2_DIR, 'frozen_hypotheses_B2.sha256')
ASAP_DIR = '/nas/data_2/wanhongz/ASAP'
DIAG_REF = '8ebbb87'
DIAG_DIR = 'experiments/2026-10-25-asap-dgm4-6class-decoder-diag'
META_DIR = '/nas/data_2/wanhongz/datasets/DGM4/metadata'
PKL = {'val': '/nas/data_2/wanhongz/failure_analysis/preds_val_all.pkl',
       'test': '/nas/data_2/wanhongz/failure_analysis/preds_test_all.pkl'}
FEATURES = ['fake_word_ratio', 'n_fake_words', 'sentence_length', 'first_fake_rel_pos']

sys.path.insert(0, ASAP_DIR)
from dataset.utils import pre_caption  # noqa: E402


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def git_show(path):
    return subprocess.run(['git', '-C', REPO, 'show', f'{DIAG_REF}:{DIAG_DIR}/{path}'],
                          capture_output=True, check=True).stdout


def check_frozen():
    sha = sha256_bytes(open(FROZEN, 'rb').read())
    rec = open(FROZEN_SHA).read().split()[0]
    if sha != rec:
        sys.exit(f'[abort] frozen_hypotheses_B2.json sha256={sha} 与记录 {rec} 不一致')
    print(f'[frozen] sha256={sha}')
    return json.load(open(FROZEN)), sha


def check_input(name, data, want):
    got = sha256_bytes(data)
    if got != want:
        sys.exit(f'[abort] {name} sha256={got} 与冻结记录 {want} 不一致')


def load(cfg, split):
    op = cfg['operationalization（补充，不改上面措辞）']['inputs']
    base = 'val' if split == 'val_holdout' else 'test'
    raw = open(PKL[base], 'rb').read()
    check_input(PKL[base], raw, op['preds_sha256'][base])
    rows = pickle.loads(raw)
    meta = json.load(open(f'{META_DIR}/{base}.json'))

    e01_name = f'E01_{split}_preds9.csv.gz'
    e01_raw = git_show(f'results/E01/{e01_name}')
    check_input(e01_name, e01_raw, op['A_tuned_sha256'][e01_name])
    e01 = pd.read_csv(io.BytesIO(e01_raw), compression='gzip')[['meta_index', 'y9', 'A_tuned']]

    keep = None
    if split == 'val_holdout':
        sp_raw = git_show('results/split_val.json')
        check_input('split_val.json', sp_raw, op['val_holdout'].split('sha256 ')[-1])
        keep = set(json.loads(sp_raw)['val_holdout'])

    recs = []
    for r in rows:
        mi = r['meta_index']
        if keep is not None and mi not in keep:
            continue
        a = meta[mi]
        words = pre_caption(a['text'], 50).split(' ')
        recs.append(dict(meta_index=mi, id=a['id'], fake_cls=r['fake_cls'],
                         p_TS=float(r['prob_mc'][2]), p_TA=float(r['prob_mc'][3]),
                         t_TS=int(r['target_mc'][2]), t_TA=int(r['target_mc'][3]),
                         n_words=len(words), n_bert_words=int(r['tok_word_id'].max()) + 1,
                         gt_words=list(r['gt_fake_words']), pred_words=list(r['pred_fake_words']),
                         tok_label_words=sorted(set(r['tok_word_id'][r['tok_label'] == 1].tolist()))))
    df = pd.DataFrame(recs).merge(e01, on='meta_index', how='left', validate='one_to_one')
    assert df.A_tuned.notna().all(), 'E01 A_tuned 缺行'
    if keep is not None:
        assert len(df) == len(keep), (len(df), len(keep))
    return df


def auc(score, y):
    """Mann-Whitney AUC，y=1 为正类，平局记 0.5。"""
    s = pd.Series(np.asarray(score, float)); y = np.asarray(y, int)
    n1, n0 = int(y.sum()), int(len(y) - y.sum())
    if n1 == 0 or n0 == 0:
        return np.nan
    r = s.rank().values
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


class ClusterBoot:
    """以 id 为簇的 percentile bootstrap；所有统计量共用同一组重采样。"""

    def __init__(self, ids, B, seed):
        codes, uniq = pd.factorize(pd.Series(ids))
        order = np.argsort(codes, kind='stable')
        bounds = np.searchsorted(codes[order], np.arange(len(uniq) + 1))
        self.members = [order[bounds[k]:bounds[k + 1]] for k in range(len(uniq))]
        rng = np.random.default_rng(seed)
        self.draws = [rng.integers(0, len(uniq), len(uniq)) for _ in range(B)]

    def ci(self, stat):
        vals = []
        for d in self.draws:
            idx = np.concatenate([self.members[k] for k in d])
            vals.append(stat(idx))
        vals = np.asarray(vals, float)
        vals = vals[~np.isnan(vals)]
        return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def holm(p):
    p = np.asarray(p, float); m = len(p); order = np.argsort(p)
    adj = np.empty(m); run = 0.0
    for k, i in enumerate(order):
        run = max(run, (m - k) * p[i]); adj[i] = min(1.0, run)
    return adj


def verdict_t1(a):
    return '支持' if a >= 0.90 else ('证据不足' if a >= 0.75 else '不支持')


def verdict_t2(a, p):
    if a >= 0.60:
        return '方向相反' if p < 0.05 else '证据不足'
    if a <= 0.40:
        return '支持' if p < 0.05 else '证据不足'
    return '证据不足' if a <= 0.45 else '不支持'


def verdict_t3(x):
    return '支持' if x >= 0.30 else ('不支持' if x <= 0.10 else '证据不足')


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', required=True, choices=['val_holdout', 'test'])
    ap.add_argument('--confirm_test', action='store_true')
    a = ap.parse_args(argv)
    if a.split == 'test' and not a.confirm_test:
        sys.exit('[abort] test 只在 val_holdout 结果经确认后运行一次；需加 --confirm_test')
    cfg, fsha = check_frozen()
    bs = cfg['operationalization（补充，不改上面措辞）']['bootstrap']
    out = os.path.join(B2_DIR, 'val' if a.split == 'val_holdout' else 'test')
    os.makedirs(out, exist_ok=True)

    df = load(cfg, a.split)
    df['text_tampered'] = ((df.t_TS == 1) | (df.t_TA == 1)).astype(int)
    df['s'] = np.maximum(df.p_TS, df.p_TA)
    df['A_txt_none'] = df.A_tuned.isin([0, 1, 2]).astype(int)
    boot = ClusterBoot(df.id.values, bs['B'], bs['seed'])

    # 对齐自检（冻结文件 H-T3.alignment_check_step0 要求运行时重复）
    T = df[df.text_tampered == 1]
    align = dict(n_all=len(df), n_word_count_mismatch=int((df.n_words != df.n_bert_words).sum()),
                 n_text_tampered=len(T), n_gt_vs_toklabel_mismatch=int((T.gt_words != T.tok_label_words).sum()),
                 n_empty_gt=int((T.gt_words.str.len() == 0).sum()))

    # ---------- H-T1 ----------
    y, s = df.text_tampered.values, df.s.values
    a1 = auc(s, y); lo, hi = boot.ci(lambda i: auc(s[i], y[i]))
    t1 = pd.DataFrame([dict(n_pos=int(y.sum()), n_neg=int(len(y) - y.sum()), n_ids=len(boot.members),
                            auc=a1, ci_lo=lo, ci_hi=hi, verdict=verdict_t1(a1))])
    neg, pos = s[y == 0], s[y == 1]
    tpr = []
    for f in [0.01, 0.05, 0.10]:
        thr = float(np.quantile(neg, 1 - f))
        tpr.append(dict(target_fpr=f, threshold=thr, actual_fpr=float((neg > thr).mean()), tpr=float((pos > thr).mean())))
    tpr = pd.DataFrame(tpr)

    # ---------- H-T2 ----------
    T = df[df.text_tampered == 1].reset_index(drop=True)
    T['miss'] = T.A_txt_none
    T['n_fake_words'] = T.gt_words.str.len()
    T['sentence_length'] = T.n_words
    T['fake_word_ratio'] = T.n_fake_words / T.n_words
    T['first_fake_rel_pos'] = T.gt_words.apply(min) / T.n_words
    tb = ClusterBoot(T.id.values, bs['B'], bs['seed'])
    m = T.miss.values
    rows = []
    for f in FEATURES:
        x = T[f].values.astype(float)
        p = mannwhitneyu(x[m == 1], x[m == 0], alternative='two-sided').pvalue
        lo, hi = tb.ci(lambda i: auc(x[i], m[i]))
        rows.append(dict(feature=f, n_miss=int(m.sum()), n_detected=int(len(m) - m.sum()),
                         mean_miss=float(x[m == 1].mean()), mean_detected=float(x[m == 0].mean()),
                         median_miss=float(np.median(x[m == 1])), median_detected=float(np.median(x[m == 0])),
                         auc=auc(x, m), ci_lo=lo, ci_hi=hi, p_mwu=float(p)))
    t2 = pd.DataFrame(rows)
    t2['p_holm'] = holm(t2.p_mwu)
    t2['verdict'] = [verdict_t2(r.auc, r.p_mwu) for r in t2.itertuples()]
    t2['verdict_holm'] = [verdict_t2(r.auc, r.p_holm) for r in t2.itertuples()]

    # ---------- H-T3 ----------
    T['hit'] = [int(len(set(g) & set(p)) > 0) for g, p in zip(T.gt_words, T.pred_words)]
    T['n_pred_words'] = T.pred_words.str.len()
    h = T.hit.values
    rows = []
    for name, sel in [('miss', m == 1), ('detected', m == 0)]:
        idx = np.where(sel)[0]
        lo, hi = tb.ci(lambda i: h[i][m[i] == (1 if name == 'miss' else 0)].mean())
        rows.append(dict(group=name, n=len(idx), n_hit=int(h[idx].sum()), frac_hit=float(h[idx].mean()), ci_lo=lo, ci_hi=hi,
                         frac_any_pred_word=float((T.n_pred_words.values[idx] > 0).mean()),
                         verdict=verdict_t3(float(h[idx].mean())) if name == 'miss' else '仅作参照'))
    t3 = pd.DataFrame(rows)

    # ---------- 输出 ----------
    miss_rate = dict(n_text_tampered=len(T), n_miss=int(m.sum()), miss_rate=float(m.mean()),
                     by_cls={c: dict(n=int((T.fake_cls == c).sum()), miss_rate=float(T.miss[T.fake_cls == c].mean()))
                             for c in sorted(T.fake_cls.unique())})
    for name, d in [('h_t1_auc', t1), ('h_t1_tpr_at_fpr', tpr), ('h_t2_features', t2), ('h_t3_hit', t3)]:
        d.insert(0, 'split', a.split); d['frozen_sha256'] = fsha
        d.to_csv(os.path.join(out, f'{name}.csv'), index=False)
    T[['meta_index', 'id', 'fake_cls', 'A_tuned', 'miss', 's', *FEATURES, 'hit', 'n_pred_words']].assign(
        gt=T.gt_words.apply(lambda v: ' '.join(map(str, v))), pred=T.pred_words.apply(lambda v: ' '.join(map(str, v)))
    ).to_csv(os.path.join(out, 'per_sample_text_tampered.csv'), index=False)
    summary = dict(split=a.split, frozen_sha256=fsha, alignment_check=align, miss=miss_rate,
                   bootstrap=bs, diag_ref=DIAG_REF)
    json.dump(summary, open(os.path.join(out, 'summary.json'), 'w'), ensure_ascii=False, indent=1)

    pd.set_option('display.width', 250); pd.set_option('display.max_columns', 30)
    print('[align]', align); print('[miss]', json.dumps(miss_rate, ensure_ascii=False))
    print(t1.drop(columns=['split', 'frozen_sha256']).to_string(index=False))
    print(tpr.drop(columns=['split', 'frozen_sha256']).to_string(index=False))
    print(t2.drop(columns=['split', 'frozen_sha256']).to_string(index=False))
    print(t3.drop(columns=['split', 'frozen_sha256']).to_string(index=False))
    print(f'[done] {out}  frozen_sha256={fsha}')


if __name__ == '__main__':
    main(sys.argv[1:])
