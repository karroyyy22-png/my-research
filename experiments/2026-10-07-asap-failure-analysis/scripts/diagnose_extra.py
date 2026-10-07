"""补充诊断（只用 CPU），依赖 analyze_failures.py 的输出 report/per_sample_box.csv。

1. orig 误报来源：多标签最高维、假 token、按图片来源/人脸数/最大脸尺寸分桶的误报率；
2. "框落在空处"与检测结果的交叉表；
3. 口径核对："总选第一个 mtcnn 框"的正确率。
用法: python diagnose_extra.py [--out_dir report]
"""
import os, sys, json, argparse
import numpy as np
import pandas as pd

from analyze_failures import FA_DIR, META_DIR, DIMS, SIZE_EDGES, SIZE_LABELS, load_preds, build_frame


def orig_frame(df):
    o = df[df.fake_cls == 'orig'].copy()
    o['false_alarm'] = o.prob_fake >= 0.5
    o['top_dim'] = np.array(DIMS)[np.argmax(o[[f'pmc_{d}' for d in DIMS]].values, axis=1)]
    o['any_dim_ge05'] = (o[[f'pmc_{d}' for d in DIMS]].values >= 0.5).any(axis=1)
    o['source'] = o.image.str.split('/').str[2]
    o['n_faces'] = [len(mt) for mt in o.mtcnn]
    o['nface_bin'] = pd.cut(o.n_faces, [-1, 0, 1, 2, 3, 4, np.inf], labels=['0', '1', '2', '3', '4', '5+']).astype(str)
    big = [np.sqrt(((mt[:, 2] - mt[:, 0]) * (mt[:, 3] - mt[:, 1])).max()) if len(mt) else np.nan for mt in o.mtcnn]
    o['max_face_size'] = big
    o['maxface_bin'] = pd.cut(o.max_face_size, SIZE_EDGES, right=False, labels=SIZE_LABELS).astype(str)
    o.loc[o.max_face_size.isna(), 'maxface_bin'] = '无脸'
    return o


def orig_tables(o):
    lab = o.false_alarm.map({True: '误报', False: '判对'})
    top = pd.crosstab(o.top_dim, lab, normalize='columns').reindex(DIMS)
    top.columns = [f'{c}_占比' for c in top.columns]
    tok = pd.DataFrame({
        'n': o.groupby(lab).size(),
        'any_pred_tok_rate': o.groupby(lab).any_pred_tok.mean(),
        'any_mlc_dim_ge05_rate': o.groupby(lab).any_dim_ge05.mean(),
    })
    ge = {d: o[f'pmc_{d}'] >= 0.5 for d in DIMS}
    img, txt = ge['FS'] | ge['FA'], ge['TS'] | ge['TA']
    side = pd.Series(np.select([img & ~txt, ~img & txt, img & txt], ['只有图像维', '只有文本维', '两侧都有'],
                               default='四维都<0.5'), index=o.index)
    dim = pd.DataFrame({**{f'{d}_ge05': o.groupby(lab)[f'pmc_{d}'].apply(lambda s: (s >= 0.5).mean()) for d in DIMS}})
    side_t = pd.crosstab(side, lab, normalize='columns').reindex(['只有图像维', '只有文本维', '两侧都有', '四维都<0.5'])
    out = {'table_D1a_orig_top_dim': top.reset_index(), 'table_D1b_orig_token_mlc': tok.reset_index(names='group'),
           'table_D1d_orig_dim_ge05': dim.reset_index(names='group'),
           'table_D1e_orig_side': side_t.reset_index(names='side')}
    for col, name in [('source', 'source'), ('nface_bin', 'nfaces'), ('maxface_bin', 'maxface')]:
        t = o.groupby(col).agg(n=('false_alarm', 'size'), n_false_alarm=('false_alarm', 'sum'),
                               false_alarm_rate=('false_alarm', 'mean')).reset_index()
        t['share_of_false_alarms'] = t.n_false_alarm / o.false_alarm.sum()
        out[f'table_D1c_orig_by_{name}'] = t
    return out


def offface_tables(ps):
    ps = ps.copy()
    ps['sel_group'] = np.select(
        [ps.sel_correct, ps.max_iou_to_any == 0, ps.max_iou_to_any < 0.5],
        ['选对', '选错_零重叠', '选错_0<IoU<0.5'], default='选错_落在别的脸上')
    ps['bin_detect'] = ps.prob_fake >= 0.5
    # 对应维：face_swap* 看 FS，face_attribute* 看 FA
    ps['img_dim_correct'] = np.where(ps.fake_cls.str.startswith('face_swap'), ps.pmc_FS, ps.pmc_FA) >= 0.5
    ps['img_dim_any'] = (ps.pmc_FS >= 0.5) | (ps.pmc_FA >= 0.5)
    summ = ps.groupby('sel_group').agg(
        n=('bin_detect', 'size'), bin_detect_rate=('bin_detect', 'mean'),
        mlc_correct_dim_rate=('img_dim_correct', 'mean'), mlc_any_img_dim_rate=('img_dim_any', 'mean')).reset_index()
    off = ps[(~ps.sel_correct) & (ps.max_iou_to_any < 0.5)]
    ct = pd.crosstab(off.bin_detect.map({True: '二分类判假', False: '二分类判真(漏检)'}),
                     off.img_dim_correct.map({True: '多标签对应维判出', False: '多标签对应维未判出'}), margins=True)
    ct_by_type = pd.crosstab([off.fake_cls.str.startswith('face_swap').map({True: 'FS类', False: 'FA类'}),
                              off.bin_detect.map({True: '二分类判假', False: '二分类判真(漏检)'})],
                             off.img_dim_correct.map({True: '对应维判出', False: '对应维未判出'}), margins=True)
    return {'table_D2a_sel_group_vs_detect': summ, 'table_D2b_offface_crosstab': ct.reset_index(),
            'table_D2c_offface_crosstab_by_type': ct_by_type.reset_index()}, len(off)


def zero_overlap_box_table(ps, rows):
    """零重叠样本（与所有 mtcnn 框 IoU=0）的预测框大小，与其他带框样本对照。"""
    nb = {r['meta_index']: r['pred_box_norm_cxcywh'] for r in rows}
    ps = ps.copy()
    wh = np.stack([nb[m][2:] for m in ps.meta_index]).astype(np.float64)
    ps['pred_w'], ps['pred_h'] = wh[:, 0], wh[:, 1]
    ps['pred_area'] = ps.pred_w * ps.pred_h          # 归一化坐标下即占全图面积比
    ps['grp'] = np.where(ps.max_iou_to_any == 0, '零重叠', np.where(ps.sel_correct, '选对', '其他选错'))
    return ps.groupby('grp').agg(
        n=('pred_area', 'size'), area_lt_1pct=('pred_area', lambda s: (s < 0.01).mean()),
        area_lt_1e_4=('pred_area', lambda s: (s < 1e-4).mean()),
        w_median=('pred_w', 'median'), h_median=('pred_h', 'median'),
        area_median=('pred_area', 'median')).reset_index()


def first_box_table(ps):
    ps = ps.copy()
    ps['first_correct'] = ps.gt_idx == 0
    ps['type'] = np.where(ps.fake_cls.str.startswith('face_swap'), 'FS含混合', 'FA含混合')
    rows = [dict(group='全部带框', n=len(ps), first_box_acc=ps.first_correct.mean(),
                 largest_face_acc=ps.largest_correct.mean(), asap_sel_acc=ps.sel_correct.mean())]
    for g in ['face_swap', 'face_attribute']:
        s = ps[ps.fake_cls == g]
        rows.append(dict(group=g, n=len(s), first_box_acc=s.first_correct.mean(),
                         largest_face_acc=s.largest_correct.mean(), asap_sel_acc=s.sel_correct.mean()))
    for g, s in ps.groupby('type'):
        rows.append(dict(group=g, n=len(s), first_box_acc=s.first_correct.mean(),
                         largest_face_acc=s.largest_correct.mean(), asap_sel_acc=s.sel_correct.mean()))
    return pd.DataFrame(rows)


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument('--pred_test', default=f'{FA_DIR}/preds_test_all.pkl')
    ap.add_argument('--meta_test', default=f'{META_DIR}/test.json')
    ap.add_argument('--out_dir', default=f'{FA_DIR}/report')
    a = ap.parse_args(argv)

    rows = load_preds(a.pred_test, 0)
    meta = json.load(open(a.meta_test))
    df = build_frame(rows, meta)
    df['image'] = [r['image'] for r in rows]
    ps = pd.read_csv(os.path.join(a.out_dir, 'per_sample_box.csv'))

    tables = orig_tables(orig_frame(df))
    t2, n_off = offface_tables(ps)
    tables.update(t2)
    tables['table_D2d_zero_overlap_pred_box'] = zero_overlap_box_table(ps, rows)
    tables['table_D3_first_box'] = first_box_table(ps)
    for k, t in tables.items():
        t.to_csv(os.path.join(a.out_dir, f'{k}.csv'), index=False, encoding='utf-8-sig')
        print(f'== {k}\n{t.round(4).to_string(index=False)}\n')
    print(f'[info] 选错且最近脸 IoU<0.5 的样本数 = {n_off}')


if __name__ == '__main__':
    main(sys.argv[1:])
