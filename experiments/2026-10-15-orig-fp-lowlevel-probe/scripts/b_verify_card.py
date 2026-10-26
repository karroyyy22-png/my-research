"""只读：从 results 下原始文件重建 README “B 阶段结果卡”中每个含数字的行，与 README 逐行比对。
B 阶段卡的边界为“### B 阶段结果卡”到下一个“### ”标题。用法: python b_verify_card.py <实验目录> [--fix]"""
import sys, os, re, json
import pandas as pd

E = sys.argv[1]
FIX = '--fix' in sys.argv
R = os.path.join(E, 'results')
rd = lambda p: pd.read_csv(os.path.join(R, p))


def n(x, d):  # 固定小数，负号用 U+2212
    s = f'{x:.{d}f}'
    if re.fullmatch(r'-0\.0*', s):
        s = s[1:]
    return s.replace('-', '−')


def pct(x, d=1):
    return n(100 * x, d) + '%'


def ci(lo, hi, d, k=1):
    return f'[{n(k * lo, d)}, {n(k * hi, d)}]'


exp = {}  # README 行前缀(用于定位) -> 期望整行
def put(key, line):
    exp[key] = line


# ---------- B0 ----------
b0 = {s: rd(f'B0/{s}/summary.csv').query("stratum=='all'").set_index('check') for s in ['val', 'test']}
def b0col(s):
    d = b0[s]; a = d.loc['same_id']
    return a
v, t = b0['val'], b0['test']
put('| 检查项 |', f"| 检查项 | val（n={v.loc['same_id','n']}；FP {v.loc['same_id','n_fp']} / TN {v.loc['same_id','n_tn']}） | "
    f"test（n={t.loc['same_id','n']}；FP {t.loc['same_id','n_fp']} / TN {t.loc['same_id','n_tn']}） |")
put('| train 中同 id |', f"| train 中同 id | {v.loc['same_id','n_hit']} | {t.loc['same_id','n_hit']} |")
put('| train 中同 image 路径 |', f"| train 中同 image 路径 | {v.loc['same_image','n_hit']} | {t.loc['same_image','n_hit']} |")
def folder(d):
    r = d.loc['same_folder','hit_rate']; return '100%' if r == 1 else pct(r, 2)
put('| 同 origin 目录', f"| 同 origin 目录（分块目录） | {folder(v)} | {folder(t)} |")
def cap(d):
    r = d.loc['same_caption']
    hi = '0' if r.diff_ci_hi == 0 else n(100 * r.diff_ci_hi, 2)
    return f"{pct(r.hit_rate, 2)}（{r.n_hit} 条）；FP−TN {n(100*r.diff_fp_minus_tn, 2)} pp [{n(100*r.diff_ci_lo, 2)}, {hi}]"
put('| caption 完全相同 |', f"| caption 完全相同 | {cap(v)} | {cap(t)} |")

# ---------- B1 H1 ----------
m = {s: rd(f'B1/{s}/matching.csv').set_index('scope').loc['full_split_covariates_only'] for s in ['val_v5', 'test']}
ss = {s: rd(f'B1/{s}/split_status.csv').set_index('feature') for s in ['val_v5', 'test']}
ntest = ss['val_v5'].n_features_tested.iloc[0]
put('匹配对：', f"匹配对：val {int(m['val_v5'].n_matched)} / {int(m['val_v5'].n_fp)} FP（{pct(m['val_v5'].match_rate)}），"
    f"test {int(m['test'].n_matched)} / {int(m['test'].n_fp)}（{pct(m['test'].match_rate)}）。实际检验 {ntest} 个特征，jpeg_q 被排除。缺失率 FP 与 TN 之间没有明显差异。")
vd = pd.read_csv(os.path.join(R, 'B1_verdict/val_v5+test/h1_verdict.csv')).set_index('feature')
lab = {'insufficient': '证据不足', 'not_supported': '不支持', 'support_H1': '支持', 'opposite_to_H1': '与 H1 方向相反'}
pd_txt = {'lap_var': 'FP 更低', 'noise_sigma': 'FP 更高', 'noise_sigma_flat': 'FP 更高', 'mean_saturation': 'FP 更低',
          'near_gray': 'FP 更多', 'face_lap_var': 'FP 更低', 'face_noise_sigma': 'FP 更高'}
def auc(s, f): r = ss[s].loc[f]; return f"{n(r.auc,3)} {ci(r.auc_ci_lo, r.auc_ci_hi, 3)}"
def coef(f): r = ss['val_v5'].loc[f]; return f"{n(r.coef,3)} {ci(r.coef_ci_lo, r.coef_ci_hi, 3)}"
for f, p in pd_txt.items():
    put(f'| {f} | FP', f"| {f} | {p} | {auc('val_v5', f)} | {coef(f)} | {auc('test', f)} | {lab[vd.loc[f,'verdict']]} / {lab[vd.loc[f,'verdict_holm']]} |")
for f in ['bytes_per_px', 'bright_mean', 'bright_std', 'face_bg_ratio']:
    band = vd.loc[f, 'ci_in_045_055'].replace('val=', '').replace(';test=', ' / ')
    put(f'| {f} | 0', f"| {f} | {auc('val_v5', f)} | {coef(f)} | {auc('test', f)} | {vd.loc[f,'verdict']} / {vd.loc[f,'verdict_holm']} | {band} |")
put('判定规则：', None)  # 只检查 lap_var 示例数字
ex = n(ss['val_v5'].loc['lap_var', 'auc'], 3)
sec = rd('B1/val_v5/a_secondary_matching.csv').set_index('scope')
tw = {s: rd(f'B1/{s}/twins_summary.csv').n_pairs.iloc[0] for s in ['val_v5', 'test']}
put('- 孪生组：', f"- 孪生组：val {tw['val_v5']} 对，test {tw['test']} 对，见 `twins_summary.csv`。")
put('- val 次要匹配：', f"- val 次要匹配：usa_today {pct(sec.loc['secondary(source,W,H):usa_today','match_rate'])}、"
    f"washington_post {pct(sec.loc['secondary(source,W,H):washington_post','match_rate'])}，合并 {int(sec.loc['secondary(source,W,H):pooled','n_matched'])} 对，"
    f"见 `a_secondary_matched.csv`。test 上没有匹配率低于 70% 的来源。")
ewh = {s: rd(f'B1/{s}/e_wh_source.csv').coverage_unaltered.iloc[0] for s in ['val_v5', 'test']}
assert len(rd('B1/test/a_secondary_matching.csv')) == 0

# ---------- H2 ----------
h2 = {s: rd(f'B1/{s}/h2_face_bg_ratio.csv') for s in ['val_v5', 'test']}
def h2r(s, k): r = h2[s][h2[s].comparison.str.startswith(k)].iloc[0]; return r
for k, name, pr in [('H2-1', 'H2-1 Original FP vs 配对 TN', '<0.5'), ('H2-2', 'H2-2 FS+FA vs 未改图', '<0.5'), ('H2-3', 'H2-3 漏检 FA vs 检出 FA', '>0.5')]:
    a, b = h2r('val_v5', k), h2r('test', k)
    conf = '确认' if b.dir_as_predicted and b.ci_excl_05 else '未确认'
    put(f'| {k} ', f"| {name} | {pr} | {n(a.auc,3)} {ci(a.ci_lo,a.ci_hi,3)} | {n(b.auc,3)} {ci(b.ci_lo,b.ci_hi,3)} | {conf} |")
h21t = n(h2r('test', 'H2-1').auc, 3)
adj = {s: rd(f'B1/{s}/h2_adjusted_supplement.csv') for s in ['val_v5', 'test']}
def adjs(s, k): r = adj[s][adj[s].comparison.str.startswith(k)].iloc[0]; return f"{n(r.coef_adj,3)} {ci(r.ci_lo,r.ci_hi,3)}"
put('- H2-2 系数：', f"- H2-2 系数：val {adjs('val_v5','H2-2')}，test {adjs('test','H2-2')}。")
put('- H2-3 系数：', f"- H2-3 系数：val {adjs('val_v5','H2-3')}，test {adjs('test','H2-3')}。")

# ---------- V0 ----------
v0 = {s: rd(f'B1_V0/{s}/V0_tests.csv') for s in ['val', 'test']}
def v0s(s, k):
    r = v0[s][v0[s].test.str.startswith(k)].iloc[0]
    return f"{r.n_fp_with_sib} / {r.n_tn_with_sib}；{pct(r.rate_fp)} vs {pct(r.rate_tn)}；+{n(100*r['diff'],1)} pp {ci(r.ci_lo,r.ci_hi,1,100)}"
va = v0['val'][v0['val'].test.str.startswith('A')].verdict.iloc[0]; ta = v0['test'][v0['test'].test.str.startswith('A')].verdict.iloc[0]
put('| A 组 vs TN', f"| A 组 vs TN：同图兄弟的图像维报警率（确认性） | {v0s('val','A')} | {v0s('test','A')} | val {va}，test {ta} |")
put('| B 组 vs TN', f"| B 组 vs TN：同句兄弟的文本维报警率（仅描述） | {v0s('val','B')} | {v0s('test','B')} | — |")
cov = {s: rd(f'B1_V0/{s}/V0_coverage_by_group.csv').set_index('grp') for s in ['val', 'test']}
put('- A 组有同图兄弟', f"- A 组有同图兄弟的比例：val {pct(cov['val'].loc['A','has_img_sib'])}，test {pct(cov['test'].loc['A','has_img_sib'])}。")
ph = {s: rd(f'B1_V0/{s}/posthoc_A_by_sib_cls.csv').set_index('sibling_class') for s in ['val', 'test']}
pp = lambda s, c: '+' + n(100 * ph[s].loc[c, 'diff'], 1)
put('- 事后描述', f"- 事后描述（按兄弟类别拆分 A 组差值）：text_swap 兄弟 val {pp('val','text_swap')} pp、test {pp('test','text_swap')} pp；"
    f"text_attribute 兄弟 val {pp('val','text_attribute')} pp、test {pp('test','text_attribute')} pp。见 `posthoc_A_by_sib_cls.csv`。")

# ---------- V1 ----------
d = {s: rd(f'B1_V1/{s}/d_coverage.csv').set_index('subset') for s in ['val', 'test']}
def three(s, c): return ' / '.join(pct(d[s].loc[k, c]) for k in ['all', 'FP', 'TN'])
put('| 有同句兄弟', f"| 有同句兄弟的 orig 比例（全部 / FP / TN） | {three('val','frac_has_caption_sib')} | {three('test','frac_has_caption_sib')} |")
put('| 有同图兄弟', f"| 有同图兄弟的 orig 比例（全部 / FP / TN） | {three('val','frac_has_image_sib')} | {three('test','frac_has_image_sib')} |")
A1 = {s: rd(f'B1_V1/{s}/a_caption_sib_text_dims.csv').query("stratum=='all'").set_index('dim') for s in ['val', 'test']}
B1 = {s: rd(f'B1_V1/{s}/b_image_sib_image_dims.csv').query("stratum=='all'").set_index('dim') for s in ['val', 'test']}
def a1(s): x = A1[s]; return f"{n(x.loc['TS','abs_median'],1)} / {n(x.loc['TA','abs_median'],1)}（{x.loc['TS','n_pairs']} 对）"
def b1(s): x = B1[s]; return f"{n(x.loc['FS','abs_median'],4)} / {n(x.loc['FA','abs_median'],4)}（{x.loc['FS','n_pairs']} 对）"
put('| (a) 同句兄弟', f"| (a) 同句兄弟：\\|ΔTS\\|、\\|ΔTA\\| 中位数 | {a1('val')} | {a1('test')} |")
put('| (b) 同图兄弟', f"| (b) 同图兄弟：\\|ΔFS\\|、\\|ΔFA\\| 中位数 | {b1('val')} | {b1('test')} |")
C1 = {s: rd(f'B1_V1/{s}/c_rank_caption_sib.csv').query("stratum=='all'").set_index('subset') for s in ['val', 'test']}
def c1(s, k): r = C1[s].loc[k]; return f"{pct(r.frac_sib_dim_higher)} {ci(r.ci_lo, r.ci_hi, 1, 100)}（{r.n_pairs} 对）"
put('| (c) 兄弟该维分数', f"| (c) 兄弟该维分数 > 原图的比例：全部 | {c1('val','all')} | {c1('test','all')} |")
put('| (c) 同上：原图为 FP', f"| (c) 同上：原图为 FP | {c1('val','orig_FP')} | {c1('test','orig_FP')} |")
put('| (c) 同上：原图为 TN', f"| (c) 同上：原图为 TN | {c1('val','orig_TN')} | {c1('test','orig_TN')} |")

# ---------- V2 ----------
A2 = {s: rd(f'B1_V2/{s}/a_corr.csv') for s in ['val', 'test']}
def a2(s, an, st): return A2[s][(A2[s].analysis.str.startswith(an)) & (A2[s].stratum == st)].iloc[0]
def sp(s, st, npair=True):
    r = a2(s, 'main', st); return f"{n(r.spearman,3)} {ci(r.spearman_ci_lo,r.spearman_ci_hi,3)}" + (f"（{r.n_pairs} 对）" if npair else '')
def r2(s, st): r = a2(s, 'main', st); return f"{n(r.pearson_r2,3)} {ci(r.r2_ci_lo,r.r2_ci_hi,3)}"
put('| (a) own-dim logit Spearman：全部', f"| (a) own-dim logit Spearman：全部 | {sp('val','all')} | {sp('test','all')} |")
put('| (a) Pearson R²：全部', f"| (a) Pearson R²：全部 | {r2('val','all')} | {r2('test','all')} |")
for c in ['FS', 'FA']:
    put(f'| (a) Spearman：{c}', f"| (a) Spearman：{c} 兄弟 | {sp('val',f'sib_cls={c}')} | {sp('test',f'sib_cls={c}')} |")
    put(f'| (a) Pearson R²：{c}', f"| (a) Pearson R²：{c} 兄弟 | {r2('val',f'sib_cls={c}')} | {r2('test',f'sib_cls={c}')} |")
B2 = {s: rd(f'B1_V2/{s}/b_delta_logit.csv') for s in ['val', 'test']}
def b2(s):
    r = B2[s][(B2[s].analysis == 'main') & (B2[s].stratum == 'all')].iloc[0]
    return f"{n(r['median'],2)} {ci(r.median_ci_lo,r.median_ci_hi,2)}（{n(r.q1,2)}–{n(r.q3,2)}）"
put('| (b) Δlogit', f"| (b) Δlogit 中位数 [CI]（Q1–Q3） | {b2('val')} | {b2('test')} |")
C2 = {s: rd(f'B1_V2/{s}/c_fp_subset.csv').set_index('stratum') for s in ['val', 'test']}
def c2(s): r = C2[s].loc['all']; return f"{pct(r.frac_sib_own_ge05)} {ci(r.own_ci_lo,r.own_ci_hi,1,100)}（{r.n_pairs} 对）"
put('| (c) 原图为 FP：兄弟', f"| (c) 原图为 FP：兄弟 own-dim ≥0.5 的比例 | {c2('val')} | {c2('test')} |")
def c2g(s): return ' / '.join(n(C2[s].loc[f'fp_group={g}', 'median'], 2) for g in 'ABCD')
put('| (c) 原图为 FP：Δlogit', f"| (c) 原图为 FP：Δlogit 中位数，A / B / C / D 组 | {c2g('val')} | {c2g('test')} |")
def sens(s): r = a2(s, 'sensitivity', 'all'); return f"去掉 {pct(r.frac_dropped,2)}，Spearman {n(r.spearman,3)}"
put('| 去掉截断值', f"| 去掉截断值配对后（敏感性分析） | {sens('val')} | {sens('test')} |")

# ---------- 汇总/局限 中的数字 ----------
put('- H2：三项', f"- H2：三项均确认，效应小；H2-1 的 AUC 为 {h21t}。")
drop = {s: int(m[s].n_fp - m[s].n_matched) for s in m}
put('- 主匹配丢弃', f"- 主匹配丢弃了 val {drop['val_v5']} 个、test {drop['test']} 个 FP。")
put('- 孪生组只有', f"- 孪生组只有 val {tw['val_v5']} 对、test {tw['test']} 对。")

# ---------- 比对 ----------
txt = open(os.path.join(E, 'README.md')).read()
start = txt.index('### B 阶段结果卡'); end = txt.index('\n### ', start + 1)
card = txt[start:end].split('\n')
numre = re.compile(r'(?<![\w.])[−+-]?\d+(?:\.\d+)?')
n_checked = 0; bad = []
for key, want in exp.items():
    hits = [i for i, l in enumerate(card) if l.startswith(key)]
    assert len(hits) == 1, (key, hits)
    i = hits[0]; have = card[i]
    if want is None:  # 判定规则行：只核 lap_var 示例
        n_checked += 1
        if f'val AUC {ex}' not in have: bad.append((key, have, have.replace(re.search(r'val AUC [\d.]+', have).group(), f'val AUC {ex}')))
        continue
    nh, nw = numre.findall(have), numre.findall(want)
    n_checked += len(nw)
    if have != want:
        diffs = [(a, b) for a, b in zip(nh, nw) if a != b]
        bad.append((key, have, want, diffs, len(nh) == len(nw)))
        card[i] = want
# 段落中的其他数字
extra = [(f'H2-1 的效应很小（test AUC {h21t}）', 'H2-1 test AUC'), (f'val 与 test 都是 {pct(ewh["val_v5"])}' if round(ewh['val_v5'], 3) == round(ewh['test'], 3) else 'XX', '(e) 覆盖率')]
for s, lbl in extra:
    n_checked += 1
    if s not in '\n'.join(card): bad.append((lbl, '缺失', s))
print(f'核对数字个数: {n_checked}')
print(f'不一致行数: {len(bad)}')
for b in bad:
    print('---', b[0]); print('  旧:', b[1]); print('  新:', b[2])
    if len(b) > 3: print('  数字差异(旧→新):', b[3], '' if b[4] else '[数字个数不同]')
if FIX and bad:
    open(os.path.join(E, 'README.md'), 'w').write(txt[:start] + '\n'.join(card) + txt[end:])
    print('[fixed] README 已更新')
