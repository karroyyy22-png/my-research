"""只读：逐行核对 README.md 中的每个数字。
做法：去掉反引号内的代码/路径与日期后，抽出每行的全部数字，与从来源文件重新取值并按同样格式化的期望序列比对；
有数字但没有对应检查项的行记为“未覆盖”。A 阶段文件用 git show 从 exp/6class-decoder-diag@8ebbb87 读取（不切换分支）。
用法: python verify_weakness_summary.py [README 路径，默认本实验 README.md]"""
import os, re, io, sys, json, subprocess
import pandas as pd

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = os.path.dirname(os.path.dirname(HERE))
A_REF, A_DIR = '8ebbb87', 'experiments/2026-10-25-asap-dgm4-6class-decoder-diag/results'
B_DIR = os.path.join(REPO, 'experiments/2026-10-15-orig-fp-lowlevel-probe/results')


def a_bytes(p):
    return subprocess.run(['git', '-C', REPO, 'show', f'{A_REF}:{A_DIR}/{p}'], capture_output=True, check=True).stdout


def a_csv(p):
    return pd.read_csv(io.BytesIO(a_bytes(p)))


def b_csv(p):
    return pd.read_csv(os.path.join(B_DIR, p))


def f(x, d):
    s = f'{x:.{d}f}'
    return (s[1:] if re.fullmatch(r'-0\.0*', s) else s).replace('-', '−')


pct = lambda x, d=1: f(100 * x, d)

# ---------- 来源取值 ----------
b0 = b_csv('B0/test/summary.csv').query("stratum=='all'").set_index('check').loc['same_id']
binref = a_csv('E01/E01_test_binary_refs.csv').set_index('decoder').loc['R0']
assert int(binref.orig_FP) == int(b0.n_fp) and abs(binref.orig_FPR - b0.n_fp / b0.n) < 1e-12
v0 = b_csv('B1_V0/test/V0_tests.csv'); v0 = v0[v0.test.str.startswith('A')].iloc[0]
h1 = b_csv('B1_verdict/val_v5+test/h1_verdict.csv')
dirc = h1[h1.kind == 'directed']
assert (dirc.verdict == dirc.verdict_holm).all()
undir = h1[h1.kind == 'undirected']
assert set(undir.verdict) == {'无可分辨差异'} and set(undir.verdict_holm) == {'无可分辨差异'}
assert set(h1[h1.kind == 'constant'].feature) == {'jpeg_q'}
conf = a_csv('E03/E03_test_conflict_rates.csv').set_index('cls6').loc['Original']
n_conf = int(conf['bin假_四维全<0.5_n'])
dec = a_csv('E01/E01_test_decoders_main.csv')
h2 = b_csv('B1/test/h2_face_bg_ratio.csv')
h2a = {k: h2[h2.comparison.str.startswith(k)].iloc[0] for k in ['H2-1', 'H2-2', 'H2-3']}
assert all(r.dir_as_predicted and r.ci_excl_05 for r in h2a.values())  # 三项 确认
b2s = json.load(open(os.path.join(B_DIR, 'B2/test/summary.json')))
comp = a_csv('E02/E02_axis_error_composition.csv').set_index(['split', 'decoder', 'axis'])
txt = comp.loc[('test', 'A_tuned', 'text_axis')]; img = comp.loc[('test', 'A_tuned', 'image_axis')]
assert int(txt.miss) == b2s['miss']['n_miss'] and abs(txt.miss_rate - b2s['miss']['miss_rate']) < 1e-12
t1 = b_csv('B2/test/h_t1_auc.csv').iloc[0]
t2 = b_csv('B2/test/h_t2_features.csv').set_index('feature')
assert set(t2.verdict) == {'不支持'} and set(t2.verdict_holm) == {'不支持'}
t3 = b_csv('B2/test/h_t3_hit.csv').set_index('group')
tpr = b_csv('B2/test/h_t1_tpr_at_fpr.csv').set_index('target_fpr')
axA = json.loads(a_bytes('E02/E02_test_A_tuned_axes_questions.json'))['questions']
axB = json.loads(a_bytes('E02/E02_test_B_tuned_axes_questions.json'))['questions']
ind = a_csv('E02/E02_axis_independence.csv').set_index(['split', 'decoder', 'group'])
it, iv = ind.loc[('test', 'A_tuned', 'pooled_mixed')], ind.loc[('val_holdout', 'A_tuned', 'pooled_mixed')]
lay = a_csv('E02/E02_test_A_tuned_error_layers.csv').set_index('layer')
L = lambda k: lay.loc[[i for i in lay.index if i.startswith(k)][0]]
n_err = int(lay.n.sum())

# ---------- 期望：行内唯一子串 -> 该行数字序列 ----------
E = {
 'test 中 Original 16876 张': [str(int(b0.n)), '0.5', str(int(b0.n_fp)), pct(b0.n_fp / b0.n)],
 '同一张图换成篡改文本后报警持续': [pct(v0.rate_fp), pct(v0.rate_tn), '+' + pct(v0['diff']), pct(v0.ci_lo), pct(v0.ci_hi)],
 '低层画质（H1）': [str(len(dirc)), str((dirc.verdict == 'support_H1').sum()), str((dirc.verdict == 'insufficient').sum()),
                 str((dirc.verdict == 'not_supported').sum()), str(len(undir))],
 '二分类头与四头冲突：': [str(int(binref.orig_FP)), str(n_conf), pct(n_conf / binref.orig_FP), '0.5'],
 '解码规则：9 个解码器': [str(len(dec)), f(dec.orig_FPR.min(), 3), f(dec.orig_FPR.max(), 3),
                    f(dec.macroF1_6.min(), 3), f(dec.macroF1_6.max(), 3)],
 '弱线索 H2': [f(h2a[k].auc, 3) for k in ['H2-1', 'H2-2', 'H2-3']],
 'test 中文本篡改样本': [str(b2s['miss']['n_text_tampered']), str(b2s['miss']['n_miss']), pct(b2s['miss']['miss_rate'])],
 'H-T1：文本得分 AUC': [f(t1.auc, 4), f(t1.ci_lo, 4), f(t1.ci_hi, 4)],
 'H-T2：4 个特征': [str(len(t2)), f(t2.auc.min(), 3), f(t2.auc.max(), 3)],
 'H-T3：漏检样本中命中': ['1', pct(t3.loc['miss', 'frac_hit']), pct(1 - t3.loc['miss', 'frac_any_pred_word'])],
 '阈值/校准：9 个解码器': [str(len(dec)), f(dec.ml_TS_R.min(), 3), f(dec.ml_TS_R.max(), 3),
                     f(dec.ml_TA_R.min(), 3), f(dec.ml_TA_R.max(), 3)],
 '阈值/校准（续）': ['10', pct(tpr.loc[0.10, 'tpr'])],
 '篡改量：篡改词数': [f(t2.loc['n_fake_words', 'auc'], 3), f(t2.loc['fake_word_ratio', 'auc'], 3)],
 '漏检 3433 条': [str(int(img.miss)), pct(img.miss_rate)],
 '类型混淆 1783 条': [str(int(img.type_confusion)), pct(img.type_confusion_rate)],
 '误报 3769 条': [str(int(img.false_alarm)), pct(img.false_alarm_rate)],
 'test Mixed 7468 条': [str(axA['n_mixed'])],
 'A_tuned 下 2401 条': [str(axA['mixed_to_image_only'] + axA['mixed_to_text_only']), pct(axA['mixed_to_single_rate']),
                      str(axA['mixed_to_image_only']), str(axA['mixed_to_text_only'])],
 'B_tuned 下 2169 条': [str(axB['mixed_to_image_only'] + axB['mixed_to_text_only']), pct(axB['mixed_to_single_rate']),
                      str(axB['mixed_to_image_only']), str(axB['mixed_to_text_only'])],
 '联合判对 ÷ 两轴召回之积': [f(it.joint_over_product, 3), f(it.ratio_ci_lo, 3), f(it.ratio_ci_hi, 3),
                     f(iv.joint_over_product, 3), f(iv.ratio_ci_lo, 3), f(iv.ratio_ci_hi, 3)],
 '九类错误 14150 个': [str(n_err), str(int(L('a').n)), pct(L('a').share_of_errors9), str(int(L('b').n)),
                  pct(L('b').share_of_errors9), str(int(L('c').n)), pct(L('c').share_of_errors9)],
}
assert axA['n_mixed'] == axB['n_mixed']

# ---------- 比对 ----------
NUM = re.compile(r'(?<![\w.\-−/])[+−-]?\d+(?:\.\d+)?')
lines = open(sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, 'README.md')).read().split('\n')
n_checked, bad, uncovered, used = 0, [], [], set()
for no, raw in enumerate(lines, 1):
    s = re.sub(r'`[^`]*`', '', raw)
    s = re.sub(r'\d{4}-\d{2}-\d{2}', '', s)
    got = NUM.findall(s)
    keys = [k for k in E if k in raw]
    if not got and not keys:
        continue
    if len(keys) != 1:
        (uncovered if not keys else bad).append((no, raw, got, keys)); continue
    k = keys[0]; used.add(k); want = E[k]
    n_checked += len(want)
    if got != want:
        bad.append((no, raw, got, want))
missing = [k for k in E if k not in used]
print(f'核对数字个数: {n_checked}')
print(f'不一致行数: {len(bad)}  未覆盖的含数字行: {len(uncovered)}  文档中找不到的检查项: {len(missing)}')
for no, raw, got, want in bad:
    print(f'--- 第 {no} 行: {raw.strip()[:80]}'); print('    文档:', got); print('    来源:', want)
for no, raw, got, _ in uncovered:
    print(f'--- 未覆盖 第 {no} 行: {raw.strip()[:80]}  数字 {got}')
for k in missing:
    print('--- 找不到检查项:', k)
