"""只读：从 results/B2/{val,test}/ 的原始输出重建 README “B2 结果卡”中每个含数字的行，与 README 逐行比对。
用法: python b2_verify_card.py [--fix]   (--fix 只改 README 中不一致的行)"""
import os, sys, re, json, subprocess
import pandas as pd

EXP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(EXP_DIR, 'results', 'B2')
FIX = '--fix' in sys.argv
S = {'val': 'val_holdout', 'test': 'test'}


def n(x, d):  # 固定小数，负号用 U+2212
    s = f'{x:.{d}f}'
    return (s[1:] if re.fullmatch(r'-0\.0*', s) else s).replace('-', '−')


def pct(x, d=1):
    return n(100 * x, d) + '%'


def rd(split, f):
    return pd.read_csv(os.path.join(R, split, f))


summ = {s: json.load(open(os.path.join(R, s, 'summary.json'))) for s in S}
t1 = {s: rd(s, 'h_t1_auc.csv').iloc[0] for s in S}
tpr = {s: rd(s, 'h_t1_tpr_at_fpr.csv').set_index('target_fpr') for s in S}
t2 = {s: rd(s, 'h_t2_features.csv').set_index('feature') for s in S}
t3 = {s: rd(s, 'h_t3_hit.csv').set_index('group') for s in S}
fsha = open(os.path.join(R, 'frozen_hypotheses_B2.sha256')).read().split()[0]
for s in S:
    assert summ[s]['frozen_sha256'] == fsha and summ[s]['split'] == S[s]
for c in ['5f12d72', '1abab3a', '8ebbb87']:
    subprocess.run(['git', '-C', EXP_DIR, 'cat-file', '-e', c + '^{commit}'], check=True)

exp = {}
put = exp.__setitem__
put('- 冻结文件 ', f"- 冻结文件 `results/B2/frozen_hypotheses_B2.json`，sha256 `{fsha}`，提交 `5f12d72`（任何 B2 分析之前）。"
    "val_holdout 结果提交 `1abab3a` 之后，test 只运行一次。")

# 样本与漏检率
def both(f):
    return f"{f('val')} | {f('test')} |"
put('| 全部样本 |', '| 全部样本 | ' + both(lambda s: str(summ[s]['alignment_check']['n_all'])))
put('| 文本篡改 / 文本无篡改 |', '| 文本篡改 / 文本无篡改 | ' + both(lambda s: f"{t1[s].n_pos} / {t1[s].n_neg}"))
put('| 漏检数（漏检率） |', '| 漏检数（漏检率） | ' + both(lambda s: f"{summ[s]['miss']['n_miss']}（{pct(summ[s]['miss']['miss_rate'])}）"))
CLS = ['text_swap', 'text_attribute', 'face_swap&text_swap', 'face_swap&text_attribute',
       'face_attribute&text_swap', 'face_attribute&text_attribute']
put('| 漏检率：TS', '| 漏检率：TS / TA / FS&TS / FS&TA / FA&TS / FA&TA | '
    + both(lambda s: ' / '.join(pct(summ[s]['miss']['by_cls'][c]['miss_rate']) for c in CLS)))
al = lambda s: summ[s]['alignment_check']
put('| 对齐自检', '| 对齐自检：词数不一致 / gt 与 tok_label 不一致 / 空 gt | '
    + both(lambda s: f"{al(s)['n_word_count_mismatch']} / {al(s)['n_gt_vs_toklabel_mismatch']} / {al(s)['n_empty_gt']}"))

# H-T1
put('| AUC [95% CI] |', '| AUC [95% CI] | ' + both(lambda s: f"{n(t1[s].auc,4)} [{n(t1[s].ci_lo,4)}, {n(t1[s].ci_hi,4)}]"))
put('| 判定 | 支持', '| 判定 | ' + both(lambda s: t1[s].verdict))
put('| TPR @ FPR', '| TPR @ FPR 1% / 5% / 10% | ' + both(lambda s: ' / '.join(pct(tpr[s].loc[f, 'tpr']) for f in [0.01, 0.05, 0.10])))
v = t1['val']
put('H-T1 在 val_holdout', f"H-T1 在 val_holdout 上的 AUC 95% CI [{n(v.ci_lo,4)}, {n(v.ci_hi,4)}] "
    f"{'跨' if v.ci_lo < 0.90 <= v.ci_hi else '不跨'} 0.90，属于边界情形。")

# H-T2
m, d = t3['val'].loc['miss', 'n'], t3['val'].loc['detected', 'n']
put('漏检 / 检出：', f"漏检 / 检出：val_holdout {t2['val'].n_miss.iloc[0]} / {t2['val'].n_detected.iloc[0]}，"
    f"test {t2['test'].n_miss.iloc[0]} / {t2['test'].n_detected.iloc[0]}。")
FEAT = [('fake_word_ratio', '篡改词占比', 3), ('n_fake_words', '篡改词数', 2),
        ('sentence_length', '句子长度', 2), ('first_fake_rel_pos', '首个篡改词相对位置', 3)]
for f, lab, dm in FEAT:
    def cell(s):
        r = t2[s].loc[f]
        return (f"{n(r.mean_miss,dm)} / {n(r.mean_detected,dm)} | {n(r.auc,4)} [{n(r.ci_lo,4)}, {n(r.ci_hi,4)}] | "
                f"{r.p_mwu:.2g} / {r.p_holm:.2g} | {r.verdict} / {r.verdict_holm}")
    put(f'| {lab} |', f"| {lab} | {cell('val')} | {cell('test')} |")

# H-T3
def hit(s, g, counts=True):
    r = t3[s].loc[g]
    return f"{pct(r.frac_hit)} [{n(100*r.ci_lo,1)}, {n(100*r.ci_hi,1)}]" + (f"（{r.n_hit} / {r.n}）" if counts else '')
put('| 漏检组命中比例', '| 漏检组命中比例 [95% CI]（命中数 / 组大小） | ' + both(lambda s: hit(s, 'miss')))
put('| 判定 | 证据不足', '| 判定 | ' + both(lambda s: t3[s].loc['miss', 'verdict']))
put('| 检出组命中比例', '| 检出组命中比例（仅作参照） | ' + both(lambda s: hit(s, 'detected')))
put('| 漏检组中 token 头', '| 漏检组中 token 头完全没有预测篡改词的比例 | '
    + both(lambda s: pct(1 - t3[s].loc['miss', 'frac_any_pred_word'])))
put('| 检出组中同一比例', '| 检出组中同一比例（仅作参照） | '
    + both(lambda s: pct(1 - t3[s].loc['detected', 'frac_any_pred_word'])))

# 判定汇总
put('- H-T1：', f"- H-T1：{t1['val'].verdict} / {t1['test'].verdict}")
v2 = {s: set(t2[s].verdict) | set(t2[s].verdict_holm) for s in S}
assert all(len(x) == 1 for x in v2.values())
put('- H-T2：', f"- H-T2：4 个特征未校正与 Holm 均为 {v2['val'].pop()} / {v2['test'].pop()}")
put('- H-T3：', f"- H-T3：{t3['val'].loc['miss','verdict']} / {t3['test'].loc['miss','verdict']}")
same = (t1['val'].verdict == t1['test'].verdict and
        (t2['val'][['verdict', 'verdict_holm']].values == t2['test'][['verdict', 'verdict_holm']].values).all() and
        t3['val'].loc['miss', 'verdict'] == t3['test'].loc['miss', 'verdict'])
put('- test 上三项判定', f"- test 上三项判定与 val_holdout {'一致' if same else '不一致'}。")

# ---------- 比对 ----------
path = os.path.join(EXP_DIR, 'README.md')
txt = open(path).read()
start = txt.index('### B2 结果卡'); end = txt.index('### 表格说明')
card = txt[start:end].split('\n')
numre = re.compile(r'(?<![\w.])[−+-]?\d+(?:\.\d+)?(?:e[−+-]?\d+)?')
n_checked, bad = 0, []
for key, want in exp.items():
    hits = [i for i, l in enumerate(card) if l.startswith(key)]
    assert len(hits) == 1, (key, hits)
    i = hits[0]; have = card[i]
    n_checked += len(numre.findall(want))
    if have != want:
        bad.append((key, have, want, [(a, b) for a, b in zip(numre.findall(have), numre.findall(want)) if a != b]))
        card[i] = want
print(f'核对数字个数: {n_checked}')
print(f'不一致行数: {len(bad)}')
for k, h, w, dd in bad:
    print('---', k); print('  旧:', h); print('  新:', w); print('  数字差异(旧→新):', dd)
if FIX and bad:
    open(path, 'w').write(txt[:start] + '\n'.join(card) + txt[end:])
    print('[fixed] README 已更新')
