"""orig 误报机制·验证①：低层图像质量（噪声/模糊/JPEG 压缩）是否驱动 orig 误报（只用 CPU）。

假设 H1：ASAP 把"图像噪声/压缩痕迹重"当成了篡改证据（尤其触发 FS/FA 维）。
依据：配对审查中 FP05 与 TN05 是同一场景、几乎相同的 caption（只差 "Monday"），
FP 版全图有明显颗粒噪声、prob_fake=0.993、FS=0.988；TN 版干净、prob_fake=0.0005。

本脚本做三件事（全部是相关性证据，因果需另做干预实验）：
  S1  orig 内部：FP vs TN 在各低层指标上的 AUC，按来源分层（来源是混杂因素）。
  S2  捷径检查：orig 与 face_swap / face_attribute 图像在同一指标上的分布差异
      （如果篡改图系统性更"脏"或更"干净"，模型就可能学到捷径）。
  S3  孪生对照：同来源同目录、caption 几乎相同、但预测一真一假的 orig 配对，
      做符号检验——FP 一侧是否系统性更噪/更糊/压缩更重。

用法:
  python probe_lowlevel.py --n_fake 5000 --workers 8
输出: <out_dir>/lowlevel_per_sample.csv 与 table_*.csv，并打印摘要。
"""
import os, sys, json, math, argparse
from multiprocessing import Pool
import numpy as np
import pandas as pd
from PIL import Image

# 复用 10-07 failure analysis 的加载函数（同一仓库内的相对路径；也可用环境变量 FA_SCRIPTS 指定）
FA_SCRIPTS = os.environ.get('FA_SCRIPTS', os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                       '../../2026-10-07-asap-failure-analysis/scripts'))
sys.path.insert(0, FA_SCRIPTS)
from analyze_failures import FA_DIR, META_DIR, DIMS, load_preds, build_frame  # noqa: E402

IMG_ROOT = os.environ.get('IMG_ROOT', '/nas/data_2/wanhongz/datasets')   # metadata 里 image 字段形如 DGM4/origin/bbc/0361/258.jpg
STD_LUMA = np.array([16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55, 14, 13, 16, 24, 40, 57, 69, 56,
                     14, 17, 22, 29, 51, 87, 80, 62, 18, 22, 37, 56, 68, 109, 103, 77, 24, 35, 55, 64, 81, 104, 113, 92,
                     49, 64, 78, 87, 103, 121, 120, 101, 72, 92, 95, 98, 112, 100, 103, 99], dtype=np.float64)
METRICS = ['noise_sigma', 'noise_sigma_face', 'lap_var', 'jpeg_q', 'bytes_per_px']


# ---------------------------------------------------------------- 单张图像指标
def immerkaer_sigma(g):
    """Immerkaer (1996) 快速噪声估计；g 为 float 灰度图。"""
    if g.shape[0] < 3 or g.shape[1] < 3:
        return np.nan
    c = (g[:-2, :-2] - 2 * g[:-2, 1:-1] + g[:-2, 2:]
         - 2 * g[1:-1, :-2] + 4 * g[1:-1, 1:-1] - 2 * g[1:-1, 2:]
         + g[2:, :-2] - 2 * g[2:, 1:-1] + g[2:, 2:])
    return math.sqrt(math.pi / 2) * np.abs(c).sum() / (6 * (g.shape[0] - 2) * (g.shape[1] - 2))


def lap_var(g):
    lap = g[:-2, 1:-1] + g[2:, 1:-1] + g[1:-1, :-2] + g[1:-1, 2:] - 4 * g[1:-1, 1:-1]
    return float(lap.var())


def jpeg_quality(im):
    """按亮度量化表相对 IJG 标准表的缩放比例反推质量（与表的排列顺序无关）。非 JPEG 返回 nan。"""
    q = getattr(im, 'quantization', None)
    if not q or 0 not in q:
        return np.nan
    scale = 100.0 * np.sum(q[0]) / STD_LUMA.sum()
    return float((200 - scale) / 2 if scale <= 100 else 5000 / scale)


def image_stats(args):
    meta_index, rel, mt = args
    path = os.path.join(IMG_ROOT, rel)
    try:
        im = Image.open(path)
        jq = jpeg_quality(im)
        g = np.asarray(im.convert('L'), dtype=np.float64)
        sig = immerkaer_sigma(g)
        sig_face = np.nan
        if len(mt):
            areas = (mt[:, 2] - mt[:, 0]) * (mt[:, 3] - mt[:, 1])
            x1, y1, x2, y2 = [int(round(v)) for v in mt[int(np.argmax(areas))]]
            crop = g[max(y1, 0):max(y2, 0), max(x1, 0):max(x2, 0)]
            sig_face = immerkaer_sigma(crop) if crop.size else np.nan
        return dict(meta_index=meta_index, noise_sigma=sig, noise_sigma_face=sig_face, lap_var=lap_var(g),
                    jpeg_q=jq, bytes_per_px=os.path.getsize(path) / g.size, err='')
    except Exception as e:  # noqa: BLE001
        return dict(meta_index=meta_index, err=repr(e)[:200])


# ---------------------------------------------------------------- 统计工具
def auc(score, y):
    """Mann-Whitney AUC：score 越大越像 y=1 时 >0.5。"""
    s = pd.Series(score).reset_index(drop=True); y = pd.Series(y).reset_index(drop=True)
    ok = s.notna(); s, y = s[ok], y[ok]
    n1, n0 = int(y.sum()), int((1 - y).sum())
    if n1 == 0 or n0 == 0:
        return np.nan
    r = s.rank()
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def sign_test_p(k, n):
    """双侧精确符号检验。"""
    if n == 0:
        return np.nan
    tail = sum(math.comb(n, i) for i in range(0, min(k, n - k) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def jaccard(a, b):
    a, b = set(a.lower().split()), set(b.lower().split())
    return len(a & b) / max(1, len(a | b))


# ---------------------------------------------------------------- 主流程
def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument('--pred_test', default=f'{FA_DIR}/preds_test_all.pkl')
    ap.add_argument('--meta_test', default=f'{META_DIR}/test.json')
    ap.add_argument('--n_fake', type=int, default=5000, help='face_swap / face_attribute 各抽多少张做捷径检查')
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--twin_jaccard', type=float, default=0.8)
    ap.add_argument('--out_dir', default=f'{FA_DIR}/lowlevel_probe')
    a = ap.parse_args(argv)
    os.makedirs(a.out_dir, exist_ok=True)

    meta = json.load(open(a.meta_test))
    df = build_frame(load_preds(a.pred_test, 0), meta)
    df['image'] = [meta[i]['image'] for i in df.meta_index]
    df['text'] = [meta[i]['text'] for i in df.meta_index]
    df['source'] = df.image.str.split('/').str[2]
    df['folder'] = df.image.str.rsplit('/', n=1).str[0]
    pm = df[[f'pmc_{d}' for d in DIMS]].values >= 0.5
    img_t, txt_t = pm[:, 0] | pm[:, 1], pm[:, 2] | pm[:, 3]
    df['fp_group'] = np.select([img_t & ~txt_t, ~img_t & txt_t, img_t & txt_t], ['A', 'B', 'C'], default='D')

    rng = np.random.default_rng(0)
    sel = [df[df.fake_cls == 'orig']]
    for c in ['face_swap', 'face_attribute']:
        sub = df[df.fake_cls == c]
        sel.append(sub.iloc[rng.choice(len(sub), min(a.n_fake, len(sub)), replace=False)])
    sub = pd.concat(sel)
    print(f'[data] 计算低层指标: {len(sub)} 张 (orig {len(sel[0])})  IMG_ROOT={IMG_ROOT}')
    first = os.path.join(IMG_ROOT, sub.image.iloc[0])
    assert os.path.exists(first), f'图像路径不存在，请改 IMG_ROOT: {first}'

    jobs = list(zip(sub.meta_index, sub.image, sub.mtcnn))
    with Pool(a.workers) as p:
        stats = pd.DataFrame(p.map(image_stats, jobs, chunksize=64))
    bad = stats[stats.err != '']
    print(f'[data] 读取失败 {len(bad)} 张', bad.head(3).to_dict('records') if len(bad) else '')
    sub = sub.merge(stats[stats.err == ''].drop(columns='err'), on='meta_index')
    keep = ['meta_index', 'fake_cls', 'source', 'folder', 'image', 'text', 'prob_fake', 'fp_group',
            *[f'pmc_{d}' for d in DIMS], *METRICS]
    sub[keep].to_csv(f'{a.out_dir}/lowlevel_per_sample.csv', index=False)

    # ---- S1 orig 内部：FP vs TN
    o = sub[sub.fake_cls == 'orig'].copy()
    o['fp'] = (o.prob_fake >= 0.5).astype(int)
    rows = []
    for m in METRICS:
        r = dict(metric=m, auc_all=auc(o[m], o.fp),
                 median_FP=o.loc[o.fp == 1, m].median(), median_TN=o.loc[o.fp == 0, m].median())
        for s in sorted(o.source.unique()):
            oo = o[o.source == s]
            r[f'auc_{s}'] = auc(oo[m], oo.fp)
        for g in 'ABCD':  # 各 FP 组 vs 全部 TN
            oo = o[(o.fp == 0) | (o.fp_group == g)]
            r[f'auc_grp{g}_vs_TN'] = auc(oo[m], oo.fp)
        r['spearman_vs_FS'] = o[m].corr(o.pmc_FS, method='spearman')
        r['spearman_vs_FA'] = o[m].corr(o.pmc_FA, method='spearman')
        rows.append(r)
    s1 = pd.DataFrame(rows)
    s1.to_csv(f'{a.out_dir}/table_S1_orig_FP_vs_TN.csv', index=False)

    # 误报率随噪声分位（来源内分位，去掉来源混杂）
    o['noise_q_in_source'] = o.groupby('source').noise_sigma.transform(
        lambda x: pd.qcut(x.rank(method='first'), 5, labels=['Q1低', 'Q2', 'Q3', 'Q4', 'Q5高']))
    s1b = o.groupby('noise_q_in_source', observed=True).agg(
        n=('fp', 'size'), fp_rate=('fp', 'mean'), FS_ge05=('pmc_FS', lambda x: (x >= .5).mean()),
        FA_ge05=('pmc_FA', lambda x: (x >= .5).mean()), TS_ge05=('pmc_TS', lambda x: (x >= .5).mean()),
        TA_ge05=('pmc_TA', lambda x: (x >= .5).mean())).reset_index()
    s1b.to_csv(f'{a.out_dir}/table_S1b_fp_rate_by_noise_quintile.csv', index=False)

    # ---- S2 捷径检查：orig vs 篡改图
    rows = []
    for c in ['face_swap', 'face_attribute']:
        t = sub[sub.fake_cls.isin(['orig', c])]
        y = (t.fake_cls == c).astype(int)
        for m in METRICS:
            rows.append(dict(contrast=f'{c} vs orig', metric=m, auc=auc(t[m], y),
                             median_fake=t.loc[y == 1, m].median(), median_orig=t.loc[y == 0, m].median()))
    s2 = pd.DataFrame(rows)
    s2.to_csv(f'{a.out_dir}/table_S2_fake_vs_orig.csv', index=False)

    # ---- S3 孪生对照
    pairs = []
    for _, g in o.groupby('folder'):
        if g.fp.nunique() < 2:
            continue
        fp_, tn_ = g[g.fp == 1], g[g.fp == 0]
        for _, f in fp_.iterrows():
            best, bj = None, a.twin_jaccard
            for _, t in tn_.iterrows():
                j = jaccard(f.text, t.text)
                if j >= bj:
                    best, bj = t, j
            if best is not None:
                pairs.append(dict(fp_meta=f.meta_index, tn_meta=best.meta_index, jaccard=bj,
                                  fp_image=f.image, tn_image=best.image, fp_prob=f.prob_fake, tn_prob=best.prob_fake,
                                  fp_group=f.fp_group,
                                  **{f'd_{m}': f[m] - best[m] for m in METRICS}))
    tw = pd.DataFrame(pairs)
    tw.to_csv(f'{a.out_dir}/table_S3_twin_pairs.csv', index=False)
    rows = []
    for m in METRICS:
        if len(tw) == 0:
            break
        d = tw[f'd_{m}'].dropna(); d = d[d != 0]
        k = int((d > 0).sum())
        rows.append(dict(metric=m, n_pairs=len(d), fp_higher=k, fp_higher_rate=k / max(1, len(d)),
                         sign_test_p=sign_test_p(k, len(d)), median_diff=d.median()))
    s3 = pd.DataFrame(rows)
    s3.to_csv(f'{a.out_dir}/table_S3_twin_sign_test.csv', index=False)

    pd.set_option('display.width', 200); pd.set_option('display.max_columns', 30)
    print('\n== S1 orig 内 FP vs TN（auc>0.5 表示指标越大越容易误报）'); print(s1.round(4).to_string(index=False))
    print('\n== S1b 来源内噪声五分位的误报率'); print(s1b.round(4).to_string(index=False))
    print('\n== S2 篡改图 vs orig（auc>0.5 表示篡改图该指标更大）'); print(s2.round(4).to_string(index=False))
    print(f'\n== S3 孪生对照：{len(tw)} 对'); print(s3.round(4).to_string(index=False))
    print(f'\n[out] {a.out_dir}')


if __name__ == '__main__':
    main(sys.argv[1:])
