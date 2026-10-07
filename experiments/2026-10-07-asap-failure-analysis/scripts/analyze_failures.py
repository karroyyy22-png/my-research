"""ASAP failure mode 分析（只用 CPU）。

用法:
  python analyze_failures.py                       # 全量，输出到 report/
  python analyze_failures.py --n 200 --out_dir report_n200   # 小样本跑通
输入: 修补后的 preds_test_all.pkl / preds_val_all.pkl（export_preds.py 导出）+ DGM4 test metadata。
"""
import os, sys, json, pickle, argparse
import numpy as np
import pandas as pd

FA_DIR = '/nas/data_2/wanhongz/failure_analysis'
META_DIR = '/nas/data_2/wanhongz/datasets/DGM4/metadata'
DIMS = ['FS', 'FA', 'TS', 'TA']
CLASSES = ['orig', 'face_swap', 'face_attribute', 'text_swap', 'text_attribute',
           'face_swap&text_swap', 'face_swap&text_attribute',
           'face_attribute&text_swap', 'face_attribute&text_attribute']
PURE_IMG = ['face_swap', 'face_attribute']
PURE_TXT = ['text_swap', 'text_attribute']
MIXED = CLASSES[5:]
NO_TEXT_TAMPER = ['orig'] + PURE_IMG
SIZE_EDGES = [0, 32, 64, 128, 256, np.inf]
SIZE_LABELS = ['<32', '32-64', '64-128', '128-256', '>=256']
NFACE_LABELS = ['2', '3', '4', '5+']


# ---------------------------------------------------------------- 加载与校验
def load_preds(path, n):
    rows = pickle.load(open(path, 'rb'))
    return rows[:n] if n > 0 else rows


def build_frame(rows, meta):
    """逐样本 DataFrame；框、token 计数等从 pkl 取，gt 框与 mtcnn 框从 metadata 取。"""
    recs = []
    for r in rows:
        m = meta[r['meta_index']]
        assert m['image'] == r['image'] and m['fake_cls'] == r['fake_cls'], f"metadata 不一致: {r['meta_index']}"
        recs.append(dict(
            meta_index=r['meta_index'], fake_cls=r['fake_cls'], W=r['W'], H=r['H'],
            prob_fake=r['prob_fake'], iou_official=r['iou_official'],
            tok_tp=r['tok_tp'], tok_fp=r['tok_fp'], tok_fn=r['tok_fn'], tok_tn=r['tok_tn'],
            any_pred_tok=len(r['pred_fake_words']) > 0,
            **{f'pmc_{d}': float(r['prob_mc'][k]) for k, d in enumerate(DIMS)},
            **{f'tmc_{d}': int(r['target_mc'][k]) for k, d in enumerate(DIMS)},
            pred_box=np.asarray(r['pred_box_px_xyxy'], dtype=np.float64),
            gt_box=np.asarray(m['fake_image_box'], dtype=np.float64) if m['fake_image_box'] else None,
            mtcnn=np.asarray(m['mtcnn_boxes'], dtype=np.float64).reshape(-1, 4),
        ))
    df = pd.DataFrame(recs)
    df['y'] = (df.fake_cls != 'orig').astype(int)
    return df


def box_valid(b, W, H):
    return b[0] < b[2] and b[1] < b[3] and b[0] >= 0 and b[1] >= 0 and b[2] <= W and b[3] <= H


def iou_one_to_many(box, boxes):
    lt = np.maximum(box[:2], boxes[:, :2])
    rb = np.minimum(box[2:], boxes[:, 2:])
    wh = np.clip(rb - lt, 0, None)
    inter = wh[:, 0] * wh[:, 1]
    a1 = (box[2] - box[0]) * (box[3] - box[1])
    a2 = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    return inter / (a1 + a2 - inter)


def check_boxes(df):
    """对带框样本做格式与 gt∈mtcnn 校验，返回 (带框子表, 校验统计 dict)。"""
    box = df[df.gt_box.notna()].copy()
    gt_bad, mt_bad, n_match = [], [], []
    for g, mt, W, H in zip(box.gt_box, box.mtcnn, box.W, box.H):
        gt_bad.append(not box_valid(g, W, H))
        mt_bad.append(any(not box_valid(b, W, H) for b in mt))
        n_match.append(int(np.sum(np.all(mt == g, axis=1))))
    box['gt_fmt_bad'], box['mtcnn_fmt_bad'], box['n_gt_match'] = gt_bad, mt_bad, n_match
    box['excluded'] = box.gt_fmt_bad | box.mtcnn_fmt_bad | (box.n_gt_match != 1)
    pix_iou = np.array([iou_one_to_many(p, g[None])[0] for p, g in zip(box.pred_box, box.gt_box)])
    # 官方 box_iou(test=True) 只看 gt 的 x1==0 就判"零框"，gt 贴左边缘且预测近零框时会把 IoU 记成 1；B 部分改用像素 IoU
    box['iou_pix'] = pix_iou
    stats = dict(
        official_iou_mismatch=int(np.sum(np.abs(pix_iou - box.iou_official.values) > 1e-4)),
        n_box=len(box), gt_fmt_bad=int(box.gt_fmt_bad.sum()), mtcnn_fmt_bad=int(box.mtcnn_fmt_bad.sum()),
        gt_match_0=int((box.n_gt_match == 0).sum()), gt_match_gt1=int((box.n_gt_match > 1).sum()),
        excluded=int(box.excluded.sum()),
        pix_vs_official_iou_maxdiff=float(np.max(np.abs(pix_iou - box.iou_official.values))) if len(box) else 0.,
        excluded_examples=box[box.excluded].meta_index.head(5).tolist(),
    )
    return box, stats


# ---------------------------------------------------------------- 阈值
def best_threshold(prob, y):
    """在 val 上找 ACC 最大的阈值（pred = prob >= t），候选为相邻唯一分数的中点；并列取最接近 0.5。"""
    u = np.unique(prob)
    cands = np.concatenate([[0.5], (u[:-1] + u[1:]) / 2])
    s = np.sort(prob)
    order = np.argsort(prob, kind='stable')
    ys = y[order]
    neg_cum = np.concatenate([[0], np.cumsum(ys == 0)])
    pos_total = ys.sum()
    pos_cum = np.concatenate([[0], np.cumsum(ys == 1)])
    k = np.searchsorted(s, cands, side='left')        # 低于阈值（判真）的样本数
    acc = (neg_cum[k] + (pos_total - pos_cum[k])) / len(y)
    best = acc.max()
    tie = np.where(np.isclose(acc, best, rtol=0, atol=1e-12))[0]
    i = tie[np.argmin(np.abs(cands[tie] - 0.5))]
    return float(cands[i]), float(acc[i])


# ---------------------------------------------------------------- A
def group_masks(cls):
    g = [(c, cls == c) for c in CLASSES]
    g += [('纯图像(FS+FA)', cls.isin(PURE_IMG)), ('纯文本(TS+TA)', cls.isin(PURE_TXT)),
          ('混合(4类)', cls.isin(MIXED)), ('全部', pd.Series(True, index=cls.index))]
    return g


def table_a1(df, thr):
    err = ((df.prob_fake >= thr).astype(int) != df.y)
    tot_err = int(err.sum())
    out = []
    for name, m in group_masks(df.fake_cls):
        n, e = int(m.sum()), int(err[m].sum())
        out.append(dict(group=name, error_type='误报' if name == 'orig' else ('误报+漏检' if name == '全部' else '漏检'),
                        n=n, n_err=e, err_rate=e / n if n else np.nan, err_share=e / tot_err if tot_err else np.nan,
                        acc=1 - e / n if n else np.nan))
    return pd.DataFrame(out)


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else np.nan
    r = tp / (tp + fn) if tp + fn else np.nan
    f = 2 * p * r / (p + r) if (tp + fp) and (tp + fn) and (p + r) else np.nan
    return p, r, f


def table_a2_overall(df):
    out = []
    for d in DIMS:
        pk, tk = df[f'pmc_{d}'] >= 0.5, df[f'tmc_{d}'] == 1
        p, r, f = prf(int((pk & tk).sum()), int((pk & ~tk).sum()), int((~pk & tk).sum()))
        out.append(dict(dim=d, acc=float((pk == tk).mean()), precision=p, recall=r, f1=f, n_pos=int(tk.sum())))
    return pd.DataFrame(out)


def table_a2_by_class(df):
    """每格=该类在该维判对的比例；*_kind 标明 target=1（1-该值=漏判率）还是 target=0（1-该值=误报率）。"""
    correct = {d: (df[f'pmc_{d}'] >= 0.5).astype(int) == df[f'tmc_{d}'] for d in DIMS}
    all4 = np.logical_and.reduce([correct[d] for d in DIMS])
    out = []
    for name, m in group_masks(df.fake_cls):
        row = dict(group=name, n=int(m.sum()))
        for d in DIMS:
            row[f'{d}_correct'] = float(correct[d][m].mean()) if m.any() else np.nan
            t = df.loc[m, f'tmc_{d}'].unique()
            row[f'{d}_kind'] = ('漏判' if t[0] == 1 else '误报') if len(t) == 1 else '漏判+误报'
        row['all4_correct'] = float(all4[m].mean()) if m.any() else np.nan
        out.append(row)
    return pd.DataFrame(out)


def table_a3(df):
    out = []
    for name, m in group_masks(df.fake_cls):
        tp, fp, fn, tn = (int(df.loc[m, c].sum()) for c in ['tok_tp', 'tok_fp', 'tok_fn', 'tok_tn'])
        p, r, f = prf(tp, fp, fn)
        no_text = name in NO_TEXT_TAMPER or name == '纯图像(FS+FA)'
        out.append(dict(group=name, n=int(m.sum()), tok_tp=tp, tok_fp=fp, tok_fn=fn, tok_tn=tn,
                        precision=np.nan if no_text else p, recall=np.nan if no_text else r,
                        f1=np.nan if no_text else f, tok_fpr=fp / (fp + tn) if fp + tn else np.nan,
                        sample_fpr=float(df.loc[m, 'any_pred_tok'].mean()) if no_text and m.any() else np.nan))
    return pd.DataFrame(out)


# ---------------------------------------------------------------- B / C
def add_box_features(box, thr):
    sel_correct, max_iou, larg_correct, larg_tie, gt_idx = [], [], [], [], []
    for p, g, mt in zip(box.pred_box, box.gt_box, box.mtcnn):
        gi = int(np.where(np.all(mt == g, axis=1))[0][0]) if np.any(np.all(mt == g, axis=1)) else -1
        ious = iou_one_to_many(p, mt)
        s = int(np.argmax(ious))
        areas = (mt[:, 2] - mt[:, 0]) * (mt[:, 3] - mt[:, 1])
        tie = int(np.sum(areas == areas.max())) > 1
        # 预测框与所有人脸都不重叠时 argmax 无意义，不能算选对
        gt_idx.append(gi); sel_correct.append(bool(ious[s] > 0) and s == gi); max_iou.append(float(ious[s]))
        larg_tie.append(tie); larg_correct.append((not tie) and int(np.argmax(areas)) == gi)
    box = box.copy()
    box['gt_idx'], box['sel_correct'], box['max_iou_to_any'] = gt_idx, sel_correct, max_iou
    box['largest_correct'], box['largest_tie'] = larg_correct, larg_tie
    gw = np.array([g[2] - g[0] for g in box.gt_box]); gh = np.array([g[3] - g[1] for g in box.gt_box])
    box['gt_size'] = np.sqrt(gw * gh)
    box['gt_area_ratio'] = gw * gh / (box.W.values * box.H.values)
    box['n_faces'] = [len(mt) for mt in box.mtcnn]
    box['missed_05'] = box.prob_fake < 0.5
    box['missed_thr'] = box.prob_fake < thr
    return box


def type_groups(cls):
    return [('FS纯', cls == 'face_swap'), ('FA纯', cls == 'face_attribute'),
            ('FS含混合', cls.str.startswith('face_swap')), ('FA含混合', cls.str.startswith('face_attribute')),
            ('全部带框', pd.Series(True, index=cls.index))]


def bucket_stats(sub):
    n = len(sub)
    wrong = sub[~sub.sel_correct]
    return dict(n=n, iou_mean=sub.iou_pix.mean(), iou_gt50=(sub.iou_pix > 0.5).mean(),
                sel_acc=sub.sel_correct.mean(), n_sel_wrong=len(wrong),
                wrong_offface_rate=(wrong.max_iou_to_any < 0.5).mean() if len(wrong) else np.nan,
                largest_face_acc=sub.largest_correct.mean(), n_largest_tie=int(sub.largest_tie.sum()),
                miss_rate_05=sub.missed_05.mean(), miss_rate_thr=sub.missed_thr.mean())


def table_b(box, bucket_col, labels):
    out = []
    for tname, tm in type_groups(box.fake_cls):
        for lab in labels:
            sub = box[tm & (box[bucket_col] == lab)] if bucket_col else box[tm]
            if bucket_col is None and lab != labels[0]:
                continue
            row = dict(type_group=tname, bucket=lab)
            row.update(bucket_stats(sub) if len(sub) else dict(n=0))
            out.append(row)
    return pd.DataFrame(out)


def assign_buckets(box):
    box = box.copy()
    box['size_bin'] = pd.cut(box.gt_size, SIZE_EDGES, right=False, labels=SIZE_LABELS).astype(str)
    q = np.quantile(box.gt_area_ratio, [0, .2, .4, .6, .8, 1])
    area_labels = [f'Q{i + 1}[{q[i]:.4f},{q[i + 1]:.4f}{"]" if i == 4 else ")"}' for i in range(5)]
    box['area_bin'] = pd.cut(box.gt_area_ratio, q, right=False, labels=area_labels, include_lowest=True).astype(str)
    box.loc[box.gt_area_ratio >= q[-1], 'area_bin'] = area_labels[-1]
    box['nface_bin'] = np.where(box.n_faces >= 5, '5+', box.n_faces.astype(str))
    return box, area_labels


# ---------------------------------------------------------------- 输出
def md_table(df, float_fmt='{:.4f}'):
    cols = list(df.columns)
    lines = ['| ' + ' | '.join(cols) + ' |', '|' + '---|' * len(cols)]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            cells.append('' if (isinstance(v, float) and np.isnan(v)) else
                         (float_fmt.format(v) if isinstance(v, (float, np.floating)) else str(v)))
        lines.append('| ' + ' | '.join(cells) + ' |')
    return '\n'.join(lines)


def write_report(out_dir, tables, info):
    os.makedirs(out_dir, exist_ok=True)
    for name, t in tables.items():
        t.to_csv(os.path.join(out_dir, f'{name}.csv'), index=False, encoding='utf-8-sig')
    L = ['# ASAP failure mode 分析（草稿）', '',
         f"数据：{info['pred_test']}（n={info['n_test']}），阈值选择用 {info['pred_val']}（n={info['n_val']}）。只用 CPU 计算。", '',
         '## 口径说明',
         f"- 二分类阈值：0.5；val 上 ACC 最大的阈值 = {info['thr']:.6f}（val ACC = {info['thr_val_acc'] * 100:.4f}）。",
         '- 官方 IoU（74.43）把非人脸样本也算进去（gt 为零框且预测接近零框时记为 1）；B 部分只统计带 fake_image_box 的样本，均值与 74.43 不同。',
         '- 官方 box_iou(test=True) 只用 gt 的 x1==0 判断零框：gt 贴图像左边缘、预测为近零框的人脸样本会被错记 IoU=1（数量见数据校验 official_iou_mismatch）。B 部分 IoU 一律用像素坐标重算的 IoU。',
         '- 候选框选择：预测框与所有 mtcnn 框 IoU 都为 0 时记为选错（且计入"框没落在任何脸上"）。',
         '- 多标签各类 F1 用多标签头口径（prob_mc>=0.5 vs target_mc，全部样本），不是 7/21 报告里按单一类子集算的二分类 F1。',
         '', '## 数据校验',
         *[f'- {k}: {v}' for k, v in info['box_check'].items()],
         f"- 全量 IoU 还原：带框样本 IoU 和 + 非带框样本 IoU 和 → 全体均值 = {info['iou_reconstruct'] * 100:.4f}",
         '']
    sections = [
        ('## A1 二分类（阈值 0.5）', 'table_A1_binary_thr0.5'),
        ('## A1 二分类（val 最优阈值）', 'table_A1_binary_thrval'),
        ('## A2 多标签头：各维总体', 'table_A2_mlc_overall'),
        ('## A2 多标签头：按类判对比例（kind=漏判 时 1-correct 为漏判率；误报 时为误报率）', 'table_A2_mlc_by_class'),
        ('## A3 文本定位（无文本篡改的组只报误报率）', 'table_A3_text_loc'),
        ('## B/C 图像定位：总体', 'table_B_overall'),
        ('## B/C 按 gt 人脸尺寸 sqrt(w·h)', 'table_B_size'),
        ('## B/C 按 gt 人脸占图面积比（五分位，边界在全部带框样本上计算）', 'table_B_area'),
        ('## B/C 按图中人脸数', 'table_B_nfaces'),
    ]
    for title, key in sections:
        L += [title, '', md_table(tables[key]), '', '结论：（待写）', '']
    L += ['## C 说明',
          '- `largest_face_acc` = 总选面积最大 mtcnn 框的选择正确率，按定义等于"gt 恰好是最大脸的比例"；最大面积并列的样本计为不正确，并列数见 `n_largest_tie`。',
          f"- 全部带框样本：gt 恰为最大脸的比例 = {info['largest_overall'] * 100:.2f}%，最大面积并列 {info['largest_tie_overall']} 条。",
          '', '## D 扰动', '- DGM4 metadata 中没有扰动相关字段，本报告不做扰动分析。', '']
    with open(os.path.join(out_dir, 'failure_report_draft.md'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(L))


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument('--pred_test', default=f'{FA_DIR}/preds_test_all.pkl')
    ap.add_argument('--pred_val', default=f'{FA_DIR}/preds_val_all.pkl')
    ap.add_argument('--meta_test', default=f'{META_DIR}/test.json')
    ap.add_argument('--n', type=int, default=0, help='>0 时只取 test 前 n 条（val 阈值仍用 val 全量）')
    ap.add_argument('--out_dir', default=f'{FA_DIR}/report')
    a = ap.parse_args(argv)

    meta = json.load(open(a.meta_test))
    df = build_frame(load_preds(a.pred_test, a.n), meta)
    val_rows = load_preds(a.pred_val, 0)
    vp = np.array([r['prob_fake'] for r in val_rows]); vy = np.array([int(r['fake_cls'] != 'orig') for r in val_rows])
    thr, thr_acc = best_threshold(vp, vy)
    print(f'[data] test n={len(df)}  val n={len(vp)}  val 最优阈值={thr:.6f}  val ACC={thr_acc * 100:.4f}')

    box, box_stats = check_boxes(df)
    print('[box check]', box_stats)
    nb = df[df.gt_box.isna()]
    iou_rec = (box.iou_official.sum() + nb.iou_official.sum()) / len(df)

    boxv = add_box_features(box[~box.excluded], thr)
    boxv, area_labels = assign_buckets(boxv)

    tables = {
        'table_A1_binary_thr0.5': table_a1(df, 0.5),
        'table_A1_binary_thrval': table_a1(df, thr),
        'table_A2_mlc_overall': table_a2_overall(df),
        'table_A2_mlc_by_class': table_a2_by_class(df),
        'table_A3_text_loc': table_a3(df),
        'table_B_overall': table_b(boxv, None, ['all']),
        'table_B_size': table_b(boxv, 'size_bin', SIZE_LABELS),
        'table_B_area': table_b(boxv, 'area_bin', area_labels),
        'table_B_nfaces': table_b(boxv, 'nface_bin', NFACE_LABELS),
    }
    info = dict(pred_test=a.pred_test, pred_val=a.pred_val, n_test=len(df), n_val=len(vp),
                thr=thr, thr_val_acc=thr_acc, box_check=box_stats, iou_reconstruct=iou_rec,
                largest_overall=float(boxv.largest_correct.mean()), largest_tie_overall=int(boxv.largest_tie.sum()))
    write_report(a.out_dir, tables, info)
    boxv.drop(columns=['pred_box', 'gt_box', 'mtcnn']).to_csv(os.path.join(a.out_dir, 'per_sample_box.csv'),
                                                             index=False, encoding='utf-8-sig')

    # 交叉核对
    t1 = tables['table_A1_binary_thr0.5']; t3 = tables['table_A3_text_loc']
    print(f"[verify] ACC@0.5 全部 = {t1[t1.group == '全部'].acc.item() * 100:.4f}")
    print(f"[verify] 多标签 F1 = {[round(x * 100, 2) for x in tables['table_A2_mlc_overall'].f1]}")
    print(f"[verify] token F1 全部 = {t3[t3.group == '全部'].f1.item() * 100:.4f}")
    print(f"[verify] IoU 全体均值还原 = {iou_rec * 100:.4f}；带框样本 IoU 均值 = {box.iou_official.mean() * 100:.4f}")
    print(f'[save] {a.out_dir}  tables={len(tables)}')


if __name__ == '__main__':
    main(sys.argv[1:])
