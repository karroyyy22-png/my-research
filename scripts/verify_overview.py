"""只读：逐行核对 PROJECT_OVERVIEW.md 与 docs/DECISIONS.md 中的每个数字。
做法同 experiments/2026-10-26-asap-dgm4-weakness-summary/scripts/verify_weakness_summary.py：
去掉反引号内的代码/路径、Markdown 链接、日期、标题行、列表序号与编号标签（弱点 N、阶段 N、#N、CVPR 年份）后，
抽出每行的全部数字，与从来源文件重新取值并按同样格式化的期望序列比对；有数字但没有对应检查项的行记为“未覆盖”。
两种取值方式：
  v —— 从结果文件（csv/json/txt）重新计算并格式化；
  t —— 结论只记录在 README 文本里（更早的实验）：确认该数字字符串原样出现在所引 README 中。
所有来源都在本仓库工作区内（A/B 阶段与弱点汇总的目录已合并到本分支）。只用 CPU，不写任何文件。
用法: python scripts/verify_overview.py"""
import os, re, json
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXP = os.path.join(REPO, 'experiments')
A = os.path.join(EXP, '2026-10-25-asap-dgm4-6class-decoder-diag/results')
B = os.path.join(EXP, '2026-10-15-orig-fp-lowlevel-probe/results')
W = os.path.join(EXP, '2026-10-26-asap-dgm4-weakness-summary')


def f(x, d):
    s = f'{x:.{d}f}'
    return (s[1:] if re.fullmatch(r'-0\.0*', s) else s).replace('-', '−')


pct = lambda x, d=1: f(100 * x, d)
csv = lambda *p: pd.read_csv(os.path.join(*p), encoding='utf-8-sig')


def t(rel, s):
    """README 文本来源：s 必须原样出现在 experiments/<rel> 中（− 与 - 视为相同）。"""
    text = open(os.path.join(EXP, rel), encoding='utf-8').read().replace('−', '-')
    assert s.replace('−', '-') in text, f'{s} 不在 {rel}'
    return s


# ---------- 更早的实验（X01–X09） ----------
x01 = json.load(open(os.path.join(EXP, '2026-06-02-asap-baseline-repro/results/results_all.txt')))
R02, R03 = '2026-06-06-omnifd-asap-fusion/README.md', '2026-06-09-lr-batchsize-tamperdetector-ablation/README.md'
R04, R05 = '2026-07-21-omnifd-local-token-fusion/README.md', '2026-09-12-omnifd-residual-diagnosis/README.md'
R06, R07 = '2026-09-13-omnifd-boxselect-eval/README.md', '2026-09-13-localization-task-nature/README.md'
x06 = csv(EXP, '2026-09-13-omnifd-boxselect-eval/results/table_E_omnifd_boxselect_recheck.csv').set_index('group').loc['全部带框']
x08 = csv(EXP, '2026-10-07-asap-failure-analysis/results/table_A1_binary_thr0.5.csv').set_index('group').loc['orig']
P0 = os.path.join(EXP, '2026-10-13-phase0-iou-eval-split/results')
ph = lambda m, p: json.load(open(os.path.join(P0, m, p, f'phase0_split_{p}.json')))
an, ac, cc, cn = ph('ASAP', 'none'), ph('ASAP', 'cscl'), ph('CSCL', 'cscl'), ph('CSCL', 'none')
gap = cc['all_IoUmean'] - an['all_IoUmean']
side_t = an['n_tampered'] / an['n_all'] * (cc['tampered_IoUmean'] - an['tampered_IoUmean'])
side_u = an['n_untampered'] / an['n_all'] * (cc['untampered_IoUmean'] - an['untampered_IoUmean'])
assert abs(side_t + side_u - gap) < 1e-9

# ---------- A / B / W 来源 ----------
b0t = csv(B, 'B0/test/summary.csv').query("stratum=='all'").set_index('check')
b0v = csv(B, 'B0/val/summary.csv').query("stratum=='all'").set_index('check')
assert all(int(d.loc[c, 'n_hit']) == 0 for d in (b0t, b0v) for c in ('same_id', 'same_image'))
binref = csv(A, 'E01/E01_test_binary_refs.csv').set_index('decoder').loc['R0']
assert int(binref.orig_FP) == int(b0t.loc['same_id', 'n_fp'])
v0 = csv(B, 'B1_V0/test/V0_tests.csv'); v0 = v0[v0.test.str.startswith('A')].iloc[0]
assert v0.verdict == '支持'
h1 = csv(B, 'B1_verdict/val_v5+test/h1_verdict.csv')
dirc, undir = h1[h1.kind == 'directed'], h1[h1.kind == 'undirected']
assert (dirc.verdict == dirc.verdict_holm).all() and set(undir.verdict) == {'无可分辨差异'}
assert set(h1[h1.kind == 'constant'].feature) == {'jpeg_q'}
h1_cnt = [str(len(dirc)), str((dirc.verdict == 'support_H1').sum()), str((dirc.verdict == 'insufficient').sum()),
          str((dirc.verdict == 'not_supported').sum())]
conf = csv(A, 'E03/E03_test_conflict_rates.csv').set_index('cls6').loc['Original']
n_conf = int(conf['bin假_四维全<0.5_n'])
dec = csv(A, 'E01/E01_test_decoders_main.csv')
assert len(dec) == 9
boot = csv(A, 'E01/E01_val_holdout_paired_bootstrap.csv').query("metric=='macroF1_6'").set_index('pair')
assert bool(boot.loc['C_LR - A_tuned', 'ci_excludes_0']) and not bool(boot.loc['B_tuned - A_tuned', 'ci_excludes_0'])
h2 = csv(B, 'B1/test/h2_face_bg_ratio.csv')
h2a = {k: h2[h2.comparison.str.startswith(k)].iloc[0] for k in ['H2-1', 'H2-2', 'H2-3']}
assert all(r.dir_as_predicted and r.ci_excl_05 for r in h2a.values())
h2s = [f(h2a[k].auc, 3) for k in ['H2-1', 'H2-2', 'H2-3']]
v1 = csv(B, 'B1_V1/test/c_rank_caption_sib.csv').query("subset=='all' and stratum=='all'").iloc[0]
v2 = {s: csv(B, f'B1_V2/{s}/a_corr.csv').query("stratum=='all'").set_index('analysis') for s in ('test', 'val')}
v2t, v2v = v2['test'].loc['main', 'pearson_r2'], v2['val'].loc['main', 'pearson_r2']
v2t_clip = v2['test'].loc[[i for i in v2['test'].index if i.startswith('sensitivity')][0], 'pearson_r2']
b2s = json.load(open(os.path.join(B, 'B2/test/summary.json')))
comp = csv(A, 'E02/E02_axis_error_composition.csv').set_index(['split', 'decoder', 'axis'])
txt, img = comp.loc[('test', 'A_tuned', 'text_axis')], comp.loc[('test', 'A_tuned', 'image_axis')]
assert int(txt.miss) == b2s['miss']['n_miss']
t1 = csv(B, 'B2/test/h_t1_auc.csv').iloc[0]
t2 = csv(B, 'B2/test/h_t2_features.csv')
assert set(t2.verdict) == {'不支持'} and set(t2.verdict_holm) == {'不支持'}
t3 = csv(B, 'B2/test/h_t3_hit.csv').set_index('group')
axA = json.load(open(os.path.join(A, 'E02/E02_test_A_tuned_axes_questions.json')))['questions']
ind = csv(A, 'E02/E02_axis_independence.csv').set_index(['split', 'decoder', 'group'])
it, iv = ind.loc[('test', 'A_tuned', 'pooled_mixed')], ind.loc[('val_holdout', 'A_tuned', 'pooled_mixed')]
lay = csv(A, 'E02/E02_test_A_tuned_error_layers.csv').set_index('layer')
lay_a = lay.loc[[i for i in lay.index if i.startswith('a')][0]]
wv = open(os.path.join(W, 'results/verify_weakness_summary.txt'), encoding='utf-8').read()
w_n, w_bad = re.search(r'核对数字个数: (\d+)', wv).group(1), re.search(r'不一致行数: (\d+)', wv).group(1)
RA = '2026-10-25-asap-dgm4-6class-decoder-diag/README.md'
RB = '2026-10-15-orig-fp-lowlevel-probe/results/large_files_manifest.txt'
RB2 = '2026-10-15-orig-fp-lowlevel-probe/README.md'
RF = '2026-10-15-orig-fp-lowlevel-probe/frozen_hypotheses.json'
R09 = '2026-10-13-phase0-iou-eval-split/README.md'

fpr = [f(dec.orig_FPR.min(), 3), f(dec.orig_FPR.max(), 3)]
mf1 = [f(dec.macroF1_6.min(), 3), f(dec.macroF1_6.max(), 3)]
v0n = ['+' + pct(v0['diff']), pct(v0.ci_lo), pct(v0.ci_hi)]
b2n = [pct(b2s['miss']['miss_rate']), f(t1.auc, 4), pct(t3.loc['miss', 'frac_hit'])]
conf_n = [str(int(binref.orig_FP)), str(n_conf), pct(n_conf / binref.orig_FP)]

# ---------- 期望：文件 -> {行内唯一子串: 该行数字序列} ----------
E = {'PROJECT_OVERVIEW.md': {
 '| X01 | 2026': [x01['val_AUC_cls'], x01['val_IOU_score']],
 '| X02 | 2026': [t(R02, '−10.84'), t(R02, '−0.30')],
 '| X03 | 2026': [t(R03, '74.4318'), t(R03, '64.3407'), t(R03, '92.0645'), t(R03, '91.4585')],
 '| X04 | 2026': [t(R04, '+0.03'), t(R04, '−0.40')],
 '| X05 | 2026': [t(R05, '0.909'), t(R05, '1.778'), t(R05, '91.569'), t(R05, '91.306'), t(R05, '91.6076')],
 '| X06 | 2026': [f(x06.omni_sel_acc, 2), f(x06.first_acc, 2), t(R06, '84.52')],
 '| X07 | 2026': [t(R07, '100'), '2', '5'] if t(R07, '2–5') else None,
 '| X08 | 2026': [pct(x08.err_share), pct(x08.err_rate)],
 '| X09 | 2026': ['+' + f(ac['all_IoUmean'] - an['all_IoUmean'], 2), f(cn['all_IoUmean'] - cc['all_IoUmean'], 2), f(gap, 2),
             f(side_u, 2), f(100 * side_u / gap, 0), f(side_t, 2), f(100 * side_t / gap, 0)],
 '| A-E01 |': [str(len(dec))] + mf1,
 '| A-E02 |': [pct(lay_a.share_of_errors9)],
 '| A-E03 |': conf_n,
 '| B0 | 2026': ['0'],
 '| H1 | 2026': h1_cnt,
 '| H2 | 2026': h2s,
 '| V0 | 2026': v0n,
 '| V1 | 2026': [pct(v1.frac_sib_dim_higher)],
 '| V2 | 2026': [f(v2t, 4), f(v2v, 3)],
 '| B2 | 2026': b2n,
 '| W | `experiments': [w_n, w_bad],
 'test 中 Original 16876 张': [str(int(b0t.loc['same_id', 'n'])), '0.5', str(int(b0t.loc['same_id', 'n_fp'])),
                             pct(b0t.loc['same_id', 'n_fp'] / b0t.loc['same_id', 'n'])],
 '同一张图换成篡改文本后报警持续': [pct(v0.rate_fp), pct(v0.rate_tn)] + v0n,
 '已排除：低层画质 H1': h1_cnt + [str(len(undir))],
 '已排除：二分类头与四头冲突': conf_n + ['0.5'],
 '已排除：解码规则（9 个解码器': [str(len(dec))] + fpr + mf1,
 '弱线索 H2': h2s,
 'test 中文本篡改样本': [str(b2s['miss']['n_text_tampered']), str(b2s['miss']['n_miss']), pct(b2s['miss']['miss_rate'])],
 'H-T1：文本得分 AUC': [f(t1.auc, 4), f(t1.ci_lo, 4), f(t1.ci_hi, 4)],
 'H-T3：漏检样本中命中': ['1', pct(t3.loc['miss', 'frac_hit']), pct(1 - t3.loc['miss', 'frac_any_pred_word'])],
 '已排除：阈值 / 校准': [str(len(dec)), f(dec.ml_TS_R.min(), 3), f(dec.ml_TS_R.max(), 3),
                     f(dec.ml_TA_R.min(), 3), f(dec.ml_TA_R.max(), 3)],
 '已排除：篡改量 H-T2': [str(len(t2)), f(t2.auc.min(), 3), f(t2.auc.max(), 3)],
 'test、A_tuned、图像轴': [str(int(img.miss)), pct(img.miss_rate), str(int(img.type_confusion)), pct(img.type_confusion_rate),
                     str(int(img.false_alarm)), pct(img.false_alarm_rate)],
 'test Mixed 7468 条': [str(axA['n_mixed']), str(axA['mixed_to_image_only'] + axA['mixed_to_text_only']),
                      pct(axA['mixed_to_single_rate']), str(axA['mixed_to_image_only']), str(axA['mixed_to_text_only'])],
 'Mixed 内“联合判对': [f(it.joint_over_product, 3), f(it.ratio_ci_lo, 3), f(it.ratio_ci_hi, 3),
                    f(iv.joint_over_product, 3), f(iv.ratio_ci_lo, 3), f(iv.ratio_ci_hi, 3)],
 'H1 的整体读法是': h1_cnt[1:],
 '>5MB 的原始 CSV': [t(RB, '超过 5MB') and '5'],
 '| FP / TN |': [t(RA, 'p_fake≥0.5') and '0.5'],
 '- 口径说明：这里的 24.1%': [pct(b0t.loc['same_id', 'n_fp'] / b0t.loc['same_id', 'n']), '0.5',
                         f(dec.set_index('decoder').loc['A_tuned', 'orig_FPR'], 3)],
 '- 关于“已排除”': [h1_cnt[0], h1_cnt[2], h1_cnt[3]],
 '- X02 README 的“融合前”': [t(R02, '−10.84')],
 '- 取整差异': [t('../README.md', '56%')[:-1], pct(x08.err_share), t('2026-10-07-asap-failure-analysis/README.md', '65.9%')[:-1],
            f(x06.first_acc, 2), t(RB2, '| 0.004 [') and '0.004', f(v2t, 4), t(R09, '84.07'), t(R09, '差 9.6 分') and '9.6',
            f(cc['all_IoUmean'], 2), f(gap, 2)],
 '| R0 / R1 |': [t(RA, '0.5 / 0.736491') and '0.5', '0.736491'],
 '| FP 分组 A/B/C/D |': [t(RF, '(FS 或 FA)>=0.5') and '0.5', '0.5', t(RF, '四维都 <0.5') and '0.5'],
 '| val_dev / val_holdout |': [t(RA, 'RandomState(2026)') and '2026'],
}, 'docs/DECISIONS.md': {
 '- 依据：9 个解码器': [str(len(dec))] + mf1,
 '共 32 次比较': [t(RA, '共 32 次比较') and '32'],
 '- 依据：V2 显示': [f(v2t, 4), f(v2v, 3), f(v2t_clip, 4)],
 '- 决定：H1 按冻结判定标签': h1_cnt[1:],
 '- 理由：H2 效应小': [h2s[0]],
}}

# ---------- 比对 ----------
NUM = re.compile(r'(?<![\w.\-−/])[+−-]?\d+(?:\.\d+)?')
LABEL = re.compile(r'弱点 ?[1-4]|阶段 ?[0-2]|#\d+|CVPR \d{4}')
n_checked, bad, uncovered, missing = 0, [], [], []
for doc, exp in E.items():
    used = set()
    for no, raw in enumerate(open(os.path.join(REPO, doc), encoding='utf-8').read().split('\n'), 1):
        if raw.startswith('#'):
            continue
        s = re.sub(r'`[^`]*`', '', raw)
        s = re.sub(r'\[[^\]]*\]\([^)]*\)', '', s)
        s = re.sub(r'\d{4}-\d{2}-\d{2}', '', s)
        s = re.sub(r'^\s*\d+\.\s', '', s)
        s = LABEL.sub('', s)
        got = NUM.findall(s)
        keys = [k for k in exp if (raw.startswith(k) if k.startswith('|') else k in raw)]
        if not got and not keys:
            continue
        if len(keys) != 1:
            (uncovered if not keys else bad).append((doc, no, raw, got, keys)); continue
        k = keys[0]; used.add(k); want = exp[k]
        n_checked += len(want)
        if got != want:
            bad.append((doc, no, raw, got, want))
    missing += [(doc, k) for k in exp if k not in used]
print(f'核对数字个数: {n_checked}')
print(f'不一致行数: {len(bad)}  未覆盖的含数字行: {len(uncovered)}  文档中找不到的检查项: {len(missing)}')
for doc, no, raw, got, want in bad:
    print(f'--- {doc} 第 {no} 行: {raw.strip()[:80]}'); print('    文档:', got); print('    来源:', want)
for doc, no, raw, got, _ in uncovered:
    print(f'--- 未覆盖 {doc} 第 {no} 行: {raw.strip()[:80]}  数字 {got}')
for doc, k in missing:
    print(f'--- 找不到检查项: {doc}: {k}')
