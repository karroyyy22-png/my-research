"""9/13 OmniFD 候选框选择评估的重算（只用 CPU，不重跑 OmniFD）。

输入：face_cropped_features/boxselect_eval.npz（eval_omnifd_boxselect.py 的输出：每个 (样本, 候选框) 的 OmniFD 伪造分数）
      allbox_eval.npz（同一模型，含 orig；只用来交叉核对伪造样本部分的分数）
两种口径：
  - 旧口径：被选框与 gt 的 IoU 均值（9/13 报告的 54.90% 即此口径）
  - 今天口径：被选框 == gt 即选对（严格选对率）；分数并列时取第一个（与 argmax 一致），并单独统计并列数
"""
import json
from collections import defaultdict
import numpy as np
import pandas as pd

ANN = '/nas/data_2/wanhongz/datasets/DGM4/metadata/test.json'
FC = '/nas/data_2/wanhongz/face_cropped_features'


def iou(a, b):
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    i = (x2 - x1) * (y2 - y1)
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i
    return i / u if u > 0 else 0.0


def per_sample(data, scores, items):
    per = defaultdict(list)
    for (si, bi), s in zip(items.tolist(), scores.tolist()):
        per[si].append((bi, s))
    rows = []
    for si, lst in per.items():
        it = data[si]
        lst.sort()
        mb, gt = it['mtcnn_boxes'], it['fake_image_box']
        sc = np.array([s for _, s in lst])
        assert [b for b, _ in lst] == list(range(len(mb))), f'候选框不完整: {si}'
        gi = [k for k, b in enumerate(mb) if b == gt]
        sel = int(sc.argmax())
        rows.append(dict(
            meta_index=si, fake_cls=it['fake_cls'], n_faces=len(mb), n_gt_match=len(gi),
            omni_sel_correct=len(gi) == 1 and sel == gi[0], omni_sel_iou=iou(gt, mb[sel]),
            omni_tie=int(np.sum(sc == sc.max())) > 1,
            first_correct=len(gi) == 1 and gi[0] == 0, first_iou=iou(gt, mb[0])))
    return pd.DataFrame(rows)


def summarize(df):
    groups = [('全部带框', df.index == df.index), ('face_swap', df.fake_cls == 'face_swap'),
              ('face_attribute', df.fake_cls == 'face_attribute'),
              ('FS含混合', df.fake_cls.str.startswith('face_swap')),
              ('FA含混合', df.fake_cls.str.startswith('face_attribute'))]
    out = []
    for name, m in groups:
        s = df[m]
        out.append(dict(group=name, n=len(s),
                        omni_sel_acc=s.omni_sel_correct.mean() * 100, omni_iou_mean=s.omni_sel_iou.mean() * 100,
                        first_acc=s.first_correct.mean() * 100, first_iou_mean=s.first_iou.mean() * 100,
                        omni_ties=int(s.omni_tie.sum())))
    return pd.DataFrame(out)


def main():
    data = json.load(open(ANN))
    z = np.load(f'{FC}/boxselect_eval.npz')
    za = np.load(f'{FC}/allbox_eval.npz')
    # 交叉核对：allbox 中伪造样本的分数应与 boxselect 一致
    amap = {tuple(k): v for k, v in zip(za['items'].tolist(), za['scores'].tolist())}
    d = np.array([abs(amap[tuple(k)] - v) for k, v in zip(z['items'].tolist(), z['scores'].tolist())])
    print(f'[check] boxselect 候选框数={len(z["scores"])}  allbox 候选框数={len(za["scores"])}  '
          f'伪造样本分数 |Δ| max={d.max():.2e}')
    # 旧脚本保存的 omni/first（按类别排序拼接）均值，应复现 9/13 的打印值
    print(f'[check] npz 内存的旧口径均值: omni={z["omni"].mean() * 100:.2f}%  first={z["first"].mean() * 100:.2f}%')

    df = per_sample(data, z['scores'], z['items'])
    print(f'[check] 样本数={len(df)}  gt 匹配数!=1 的样本={int((df.n_gt_match != 1).sum())}  '
          f'人脸数分布={df.n_faces.value_counts().sort_index().to_dict()}')
    t = summarize(df)
    print(t.round(2).to_string(index=False))
    gap = df[df.first_correct != (df.first_iou >= 0.999999)]
    print(f'[口径差] 第一个框 IoU 均值 - 严格选对率 = {(df.first_iou.mean() - df.first_correct.mean()) * 100:.4f} 个百分点；'
          f'第一个框非 gt 但 IoU>0 的样本 {int(((~df.first_correct) & (df.first_iou > 0)).sum())} 条，'
          f'它们的 IoU 之和 / N = {df.loc[~df.first_correct, "first_iou"].sum() / len(df) * 100:.4f}')
    t.to_csv('/nas/data_2/wanhongz/failure_analysis/report/table_E_omnifd_boxselect_recheck.csv',
             index=False, encoding='utf-8-sig')


if __name__ == '__main__':
    main()
