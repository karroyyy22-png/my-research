"""配对 5 / 9 / 22 / 29 误报原因验证（只用 CPU，不修改任何输入）。

核心思路：利用 DGM4 自身的数据结构做"天然对照"
  - text_swap / text_attribute 样本用的是原封不动的原图（只改了文字）；
  - face_swap / face_attribute 样本用的是原封不动的原 caption（只改了图）。
  所以同一张真实图片会和别的 caption 一起出现，同一句真实 caption 也会和别的图片一起出现。
  · 若误报"跟着图片走"：同图兄弟样本（换了 caption）上图像维 FS/FA 仍 ≥0.5 → 图像导致；
  · 若误报"跟着 caption 走"：同句兄弟样本（换了图）上文本维 TS/TA 仍 ≥0.5 → 文本导致。
  兄弟样本是数据集里本来就有的样本，ASAP 在 test 上的预测已在 pkl 里，不需要重新推理。

事先写定的判定标准（跑之前定好，避免事后解释）：
  [V0 全体] A 组误报的同图兄弟图像维报警率显著高于 TN 的同图兄弟（基线）→ A 组"由图像携带"成立；
            B 组误报的同句兄弟文本维报警率显著高于 TN 的同句兄弟 → B 组"由文本携带"成立。
  [P5] 假设：误报来自 322.jpg 的画质（颗粒噪声），而不是内容。
       成立需同时满足：(a) 322 与 324 低频内容几乎一致、差异集中在高频；(b) 322 的高频能量明显更高；
       (c) 322 的同图兄弟样本（不同 caption）FS 也高，而 324 的同图兄弟 FS 低。
  [P9] 假设：TA 误报来自负面动词 "criticised" 的词汇偏好。
       成立需同时满足：(a) train 中该词在 TA 篡改词里的出现率明显高于在 orig caption 里的出现率；
       (b) test 中含该词的 orig 样本 TA≥0.5 的比例明显高于 orig 整体；(c) 同句兄弟样本 TA 也高。
  [P22] 假设：TS 误报是因为模型把"保留人名、其余内容和画面对不上"的真实 caption 当成了换句。
       可验证部分：(a) train 中 TS 样本的篡改词标注是否普遍"保留人名、标其余"；
       (b) 同句兄弟样本 TS 是否也高（文本携带）；(c) 同图兄弟 FS 是否也高（FS 是否由图像携带）。
       "图文内容对不上"本身需要图文相似度模型，本脚本不验证，只报告。
  [P29] 假设待定（D 组）：报告同图 / 同句兄弟、预测框落点，以及 D 区域的整体规律。

用法:
  python verify_pairs.py                    # 默认路径同 10-07 failure analysis
  IMG_ROOT=/path/to/datasets python verify_pairs.py
输出: <out_dir>/ 下的 csv 与 P5 差异图，并打印报告。
"""
import os, re, sys, json, argparse
from collections import Counter, defaultdict
import numpy as np
import pandas as pd
from PIL import Image, ImageFilter

FA_SCRIPTS = os.environ.get('FA_SCRIPTS', os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                       '../../2026-10-07-asap-failure-analysis/scripts'))
sys.path.insert(0, FA_SCRIPTS)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_failures import FA_DIR, META_DIR, DIMS, load_preds, build_frame  # noqa: E402
from probe_lowlevel import immerkaer_sigma, lap_var, jpeg_quality  # noqa: E402

IMG_ROOT = os.environ.get('IMG_ROOT', '/nas/data_2/wanhongz/datasets')
MAX_WORDS = 50
PAIRS = {  # 配对号: (FP 图像, TN 图像)，来自 ASAP_Original_FP_案例审查.html
    5: ('DGM4/origin/usa_today/0010/322.jpg', 'DGM4/origin/usa_today/0010/324.jpg'),
    9: ('DGM4/origin/bbc/0509/826.jpg', 'DGM4/origin/bbc/0117/128.jpg'),
    22: ('DGM4/origin/usa_today/0412/347.jpg', 'DGM4/origin/usa_today/0445/633.jpg'),
    29: ('DGM4/origin/usa_today/0078/394.jpg', 'DGM4/origin/usa_today/0404/632.jpg'),
}
WORD_FAMILIES = {'criticis': ['criticised', 'criticized', 'criticise', 'criticize', 'criticises', 'criticizes',
                              'criticising', 'criticizing'],
                 'accused': ['accused'], 'targeted': ['targeted'], 'mistake': ['mistake']}


# ---------------------------------------------------------------- 文本工具（与 dataset/utils.py 的 pre_caption 一致）
def pre_caption(caption, max_words=MAX_WORDS, lower=True):
    c = caption.lower() if lower else caption
    c = re.sub(r"([,.'!?\"()*#:;~])", '', c).replace('-', ' ').replace('/', ' ')
    c = c.replace('<person>', 'person').replace('<PERSON>', 'person')
    c = re.sub(r"\s{2,}", ' ', c).rstrip('\n').strip(' ')
    return ' '.join(c.split(' ')[:max_words])


def cased_words(text):
    return pre_caption(text, lower=False).split(' ')


def name_mask(text, nlp=None):
    """每个词是否属于人名。有 spaCy 则用 PERSON 实体，否则退化为"非句首的大写词"（会把地名也算进去）。"""
    words = cased_words(text)
    if nlp is not None:
        doc = nlp(' '.join(words))
        person_chars = set()
        for ent in doc.ents:
            if ent.label_ == 'PERSON':
                person_chars.update(range(ent.start_char, ent.end_char))
        mask, pos = [], 0
        for w in words:
            mask.append(pos in person_chars)
            pos += len(w) + 1
        return mask
    return [i > 0 and w[:1].isupper() for i, w in enumerate(words)]


def load_nlp():
    try:
        import spacy
        return spacy.load('en_core_web_sm'), 'spaCy PERSON'
    except Exception:  # noqa: BLE001
        return None, '大写词启发式（含地名，偏粗）'


# ---------------------------------------------------------------- 图像工具
def gray(path):
    return Image.open(path).convert('L')


def face_iou(box, mt):
    if len(mt) == 0:
        return np.array([])
    lt = np.maximum(box[:2], mt[:, :2]); rb = np.minimum(box[2:], mt[:, 2:])
    wh = np.clip(rb - lt, 0, None); inter = wh[:, 0] * wh[:, 1]
    a1 = max((box[2] - box[0]) * (box[3] - box[1]), 0); a2 = (mt[:, 2] - mt[:, 0]) * (mt[:, 3] - mt[:, 1])
    return inter / np.maximum(a1 + a2 - inter, 1e-9)


def box_report(r):
    b = r.pred_box; mt = r.mtcnn
    area = max(b[2] - b[0], 0) * max(b[3] - b[1], 0) / (r.W * r.H)
    ious = face_iou(b, mt)
    if len(mt):
        areas = (mt[:, 2] - mt[:, 0]) * (mt[:, 3] - mt[:, 1]); rank = np.argsort(-areas).argsort()
        j = int(np.argmax(ious)) if len(ious) else -1
        on = f'IoU 最大的是第 {rank[j] + 1} 大的脸（共 {len(mt)} 张），IoU={ious[j]:.2f}，脸宽≈{np.sqrt(areas[j]):.0f}px'
    else:
        on = '无 mtcnn 人脸'
    return f'预测框占全图 {area * 100:.3f}%{"（近零框）" if area < 1e-4 else ""}；{on}'


# ---------------------------------------------------------------- 主流程
def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument('--pred_test', default=f'{FA_DIR}/preds_test_all.pkl')
    ap.add_argument('--meta_test', default=f'{META_DIR}/test.json')
    ap.add_argument('--meta_train', default=f'{META_DIR}/train.json')
    ap.add_argument('--out_dir', default=f'{FA_DIR}/verify_pairs')
    a = ap.parse_args(argv)
    os.makedirs(a.out_dir, exist_ok=True)
    nlp, ner_desc = load_nlp()
    out = []  # 报告行

    def say(s=''):
        print(s); out.append(s)

    meta = json.load(open(a.meta_test))
    pred_rows = load_preds(a.pred_test, 0)
    df = build_frame(pred_rows, meta)
    df['image'] = [meta[i]['image'] for i in df.meta_index]
    df['text'] = [meta[i]['text'] for i in df.meta_index]
    df['cap'] = [pre_caption(t) for t in df.text]
    pm = df[[f'pmc_{d}' for d in DIMS]].values
    df['img_alarm'] = (pm[:, 0] >= .5) | (pm[:, 1] >= .5)
    df['txt_alarm'] = (pm[:, 2] >= .5) | (pm[:, 3] >= .5)
    pw = {r['meta_index']: r['pred_fake_words'] for r in pred_rows}
    df['pred_words'] = [pw[m] for m in df.meta_index]
    orig = df.fake_cls == 'orig'
    df['grp'] = 'fake'
    df.loc[orig & (df.prob_fake < .5), 'grp'] = 'TN'
    g = np.select([df.img_alarm & ~df.txt_alarm, ~df.img_alarm & df.txt_alarm, df.img_alarm & df.txt_alarm],
                  ['A', 'B', 'C'], default='D')
    df.loc[orig & (df.prob_fake >= .5), 'grp'] = pd.Series(g, index=df.index)[orig & (df.prob_fake >= .5)]

    # ---- 兄弟样本：同图（只改了文字或另一条 orig）/ 同句（只改了图或另一条 orig）
    IMG_SIB = {'orig', 'text_swap', 'text_attribute'}      # 图片未被篡改
    TXT_SIB = {'orig', 'face_swap', 'face_attribute'}       # caption 未被篡改
    by_img, by_cap = defaultdict(list), defaultdict(list)
    for i, (im, cp, fc) in enumerate(zip(df.image, df.cap, df.fake_cls)):
        if fc in IMG_SIB:
            by_img[im].append(i)
        if fc in TXT_SIB:
            by_cap[cp].append(i)

    def sib(i, kind):
        pool = by_img[df.image.iat[i]] if kind == 'img' else by_cap[df.cap.iat[i]]
        return [j for j in pool if j != i]

    say('=' * 100)
    say('V0 全体：误报是"跟着图片走"还是"跟着 caption 走"（兄弟样本均为 test 中已有样本，未做任何改动）')
    recs = []
    for i in np.where(orig)[0]:
        si, st = sib(i, 'img'), sib(i, 'txt')
        recs.append(dict(i=i, grp=df.grp.iat[i], n_img_sib=len(si), n_txt_sib=len(st),
                         img_sib_img_alarm=np.mean(df.img_alarm.values[si]) if si else np.nan,
                         txt_sib_txt_alarm=np.mean(df.txt_alarm.values[st]) if st else np.nan,
                         img_sib_prob=np.mean(df.prob_fake.values[si]) if si else np.nan,
                         txt_sib_prob=np.mean(df.prob_fake.values[st]) if st else np.nan))
    sv = pd.DataFrame(recs)
    v0 = sv.groupby('grp').agg(
        n=('i', 'size'),
        有同图兄弟=('n_img_sib', lambda x: (x > 0).mean()),
        同图兄弟_图像维报警率=('img_sib_img_alarm', 'mean'),
        有同句兄弟=('n_txt_sib', lambda x: (x > 0).mean()),
        同句兄弟_文本维报警率=('txt_sib_txt_alarm', 'mean')).reindex(['A', 'B', 'C', 'D', 'TN'])
    say(v0.round(4).to_string())
    say('解读：A 行"同图兄弟_图像维报警率"应远高于 TN 行；B 行"同句兄弟_文本维报警率"应远高于 TN 行。')
    sv.to_csv(f'{a.out_dir}/V0_sibling_per_orig.csv', index=False)
    v0.to_csv(f'{a.out_dir}/V0_sibling_summary.csv')

    def pair_rows(k):
        res = []
        for img in PAIRS[k]:
            cand = df[(df.image == img) & orig]
            if len(cand) == 0:
                say(f'  !! 找不到 orig 样本: {img}'); res.append(None); continue
            res.append(cand.sort_values('prob_fake', ascending=(img == PAIRS[k][1])).index[0])
        return res

    def describe(i, label):
        r = df.loc[i]
        say(f'  [{label}] {r.image}  meta_index={r.meta_index}  {r.W}x{r.H}  prob_fake={r.prob_fake:.4f}  '
            f'FS/FA/TS/TA={r.pmc_FS:.3f}/{r.pmc_FA:.3f}/{r.pmc_TS:.3f}/{r.pmc_TA:.3f}')
        say(f'      caption: {r.text}')
        words = r.cap.split(' ')
        say(f'      预测为假的词: {[words[w] for w in r.pred_words if w < len(words)]}')
        say(f'      {box_report(r)}')
        pos = df.index.get_loc(i)
        for kind, name, dims in (('img', '同图兄弟（图未改、文字不同）', ['FS', 'FA']),
                                 ('txt', '同句兄弟（caption 未改、图不同）', ['TS', 'TA'])):
            ss = sib(pos, kind)
            say(f'      {name}: {len(ss)} 条')
            for j in ss[:8]:
                s = df.iloc[j]
                say(f'        - {s.fake_cls:15s} prob={s.prob_fake:.4f}  ' +
                    '  '.join(f'{d}={s[f"pmc_{d}"]:.3f}' for d in dims) +
                    (f'  caption: {s.text[:80]}' if kind == 'img' else f'  image: {s.image}'))

    # ---- P5
    say('\n' + '=' * 100); say('P5  假设：误报来自 322.jpg 的画质（颗粒噪声），而不是图像内容或文字')
    fp, tn = pair_rows(5)
    for i, lab in ((fp, 'FP'), (tn, 'TN')):
        if i is not None:
            describe(i, lab)
    pa, pb = (os.path.join(IMG_ROOT, p) for p in PAIRS[5])
    if os.path.exists(pa) and os.path.exists(pb):
        A, B = Image.open(pa), Image.open(pb)
        ga, gb = A.convert('L'), B.convert('L')
        if ga.size != gb.size:
            gb = gb.resize(ga.size, Image.BICUBIC)
        fa, fb = (np.asarray(x, dtype=np.float64) for x in (ga, gb))
        la, lb = (np.asarray(x.filter(ImageFilter.GaussianBlur(2)), dtype=np.float64) for x in (ga, gb))
        ha, hb = fa - la, fb - lb
        say(f'  文件: 322 {os.path.getsize(pa)} B, JPEG Q≈{jpeg_quality(A):.0f} | 324 {os.path.getsize(pb)} B, JPEG Q≈{jpeg_quality(B):.0f}')
        say(f'  噪声σ: 322={immerkaer_sigma(fa):.2f}  324={immerkaer_sigma(fb):.2f} | Laplacian方差: 322={lap_var(fa):.0f}  324={lap_var(fb):.0f}')
        say(f'  低频(高斯σ=2)平均绝对差 = {np.abs(la - lb).mean():.2f} 灰度级（越小说明内容越一致）')
        say(f'  高频能量(std): 322={ha.std():.2f}  324={hb.std():.2f}  比值={ha.std() / max(hb.std(), 1e-9):.2f}')
        d = np.abs(la - lb); Image.fromarray(np.clip(d * 8, 0, 255).astype(np.uint8)).save(f'{a.out_dir}/P5_lowpass_diff_x8.png')
        for arr, nm in ((ha, '322'), (hb, '324')):
            Image.fromarray(np.clip(arr * 4 + 128, 0, 255).astype(np.uint8)).save(f'{a.out_dir}/P5_highpass_{nm}.png')
        say(f'  差异图已存: P5_lowpass_diff_x8.png（内容差异）, P5_highpass_322/324.png（高频纹理）')
    else:
        say(f'  !! 原图路径不存在，请设置 IMG_ROOT（当前 {IMG_ROOT}）')
    meta_tr = json.load(open(a.meta_train))
    folder = os.path.dirname(PAIRS[5][0])
    same_folder = [m for m in meta_tr if os.path.dirname(m['image']) == folder]
    say(f'  train 中同目录 {folder} 的样本: {len(same_folder)} 条，类别分布 {dict(Counter(m["fake_cls"] for m in same_folder))}')
    say(f'  train 中是否直接用到 322/324 原图: {[m["fake_cls"] for m in meta_tr if m["image"] in PAIRS[5]]}')

    # ---- P9
    say('\n' + '=' * 100); say('P9  假设：TA 误报来自负面动词 "criticised" 的词汇偏好')
    fp, tn = pair_rows(9)
    if fp is not None:
        describe(fp, 'FP')
    tr_words = Counter(); tr_ta_fake = Counter(); tr_ta_has = Counter(); n_cls = Counter(m['fake_cls'] for m in meta_tr)
    tr_orig_has = Counter()
    for m in meta_tr:
        ws = pre_caption(m['text']).split(' '); s = set(ws)
        if m['fake_cls'] == 'orig':
            tr_orig_has.update(s)
        if 'text_attribute' in m['fake_cls']:
            tr_ta_has.update(s)
            tr_ta_fake.update({ws[p] for p in m['fake_text_pos'] if p < len(ws)})
    n_ta = sum(v for k, v in n_cls.items() if 'text_attribute' in k)
    lines = []
    for fam, forms in WORD_FAMILIES.items():
        o = sum(tr_orig_has[w] for w in forms); t = sum(tr_ta_fake[w] for w in forms); th = sum(tr_ta_has[w] for w in forms)
        te = df[orig & df.cap.apply(lambda c: any(w in c.split(' ') for w in forms))]
        hit = np.mean([any(te.cap.at[i].split(' ')[p] in forms for p in te.pred_words.at[i] if p < len(te.cap.at[i].split(' ')))
                       for i in te.index]) if len(te) else np.nan
        lines.append(dict(词族=fam, train_orig含该词比例=o / n_cls['orig'], train_TA样本中该词被标为篡改词的比例=t / max(n_ta, 1),
                          train_TA样本含该词比例=th / max(n_ta, 1), test_orig含该词条数=len(te),
                          test含该词orig_TA报警率=(te.pmc_TA >= .5).mean() if len(te) else np.nan,
                          test含该词orig_误报率=(te.prob_fake >= .5).mean() if len(te) else np.nan,
                          test含该词orig_该词被预测为假的比例=hit))
    wt = pd.DataFrame(lines)
    base_ta = (df[orig].pmc_TA >= .5).mean(); base_fp = (df[orig].prob_fake >= .5).mean()
    say(wt.round(4).to_string(index=False))
    say(f'  对照：test 全部 orig 的 TA 报警率 = {base_ta:.4f}，误报率 = {base_fp:.4f}')
    lift = pd.DataFrame([dict(word=w, ta_fake_rate=c / n_ta, orig_rate=tr_orig_has[w] / n_cls['orig'],
                              lift=(c / n_ta) / max(tr_orig_has[w] / n_cls['orig'], 1e-6))
                         for w, c in tr_ta_fake.items() if c >= 30]).sort_values('lift', ascending=False)
    lift.to_csv(f'{a.out_dir}/P9_train_TA_word_lift.csv', index=False)
    say('  train 中"作为 TA 篡改词出现率 / 在 orig caption 中出现率"最高的 25 个词（≥30 次）:')
    say('  ' + ', '.join(f'{r.word}({r.lift:.1f})' for r in lift.head(25).itertuples()))
    wt.to_csv(f'{a.out_dir}/P9_word_family.csv', index=False)

    # ---- P22
    say('\n' + '=' * 100); say(f'P22 假设：TS 误报是因为模型把"保留人名、其余对不上"的真实 caption 当成了换句（人名识别：{ner_desc}）')
    fp, tn = pair_rows(22)
    if fp is not None:
        describe(fp, 'FP')
        r = df.loc[fp]; nm = name_mask(r.text, nlp); ws = r.cap.split(' ')
        pred = set(r.pred_words)
        say('      逐词（[人名]/预测假=*）: ' + ' '.join(f'{w}{"[人名]" if nm[k] else ""}{"*" if k in pred else ""}'
                                               for k, w in enumerate(ws)))
    ts = [m for m in meta_tr if m['fake_cls'] == 'text_swap']
    rng = np.random.default_rng(0)
    ts = [ts[k] for k in rng.choice(len(ts), min(3000, len(ts)), replace=False)]
    n_name = n_name_fake = n_other = n_other_fake = 0
    for m in ts:
        ws = pre_caption(m['text']).split(' '); nm = name_mask(m['text'], nlp)[:len(ws)]; fk = set(m['fake_text_pos'])
        for k in range(len(ws)):
            if nm[k]:
                n_name += 1; n_name_fake += k in fk
            else:
                n_other += 1; n_other_fake += k in fk
    say(f'  train text_swap（抽 {len(ts)} 条）篡改词标注：人名词被标为篡改 {n_name_fake / max(n_name, 1):.3f}，'
        f'其余词被标为篡改 {n_other_fake / max(n_other, 1):.3f}')
    say('  解读：若人名比例明显低于其余词，说明 TS 标签形状就是"保留人名、标其余"，配对22 的预测与之同形。')
    ts_fp = df[(df.grp.isin(['B', 'C'])) & (df.pmc_TS >= .5)]
    nm_kept = []
    for i in ts_fp.index:
        r = df.loc[i]; nm = name_mask(r.text, nlp); ws = r.cap.split(' '); pred = set(r.pred_words)
        names = [k for k in range(min(len(ws), len(nm))) if nm[k]]
        if names:
            nm_kept.append(np.mean([k not in pred for k in names]))
    say(f'  test 中 TS 触发的 orig 误报 {len(ts_fp)} 条：其中人名词未被预测为假的平均比例 = {np.mean(nm_kept):.3f}（{len(nm_kept)} 条含人名）')
    o = df[orig].copy(); o['portrait'] = o.H > o.W
    po = o.groupby('portrait').agg(n=('prob_fake', 'size'), 误报率=('prob_fake', lambda x: (x >= .5).mean()),
                                   FS报警率=('pmc_FS', lambda x: (x >= .5).mean()), TS报警率=('pmc_TS', lambda x: (x >= .5).mean()))
    say('  竖图(H>W) vs 横图 的 orig 误报率:'); say(po.round(4).to_string())

    # ---- P29
    say('\n' + '=' * 100); say('P29 D 组：四维都 <0.5、二分类判假')
    fp, tn = pair_rows(29)
    if fp is not None:
        describe(fp, 'FP')
    allbelow = (pm < .5).all(axis=1)
    reg = df[allbelow & (df.prob_fake >= .5)]
    say(f'  test 中"四维都<0.5 但二分类判假"共 {len(reg)} 条，真实类别分布: {dict(Counter(reg.fake_cls))}')
    o = df[orig].copy()
    o['maxface'] = [np.sqrt(((m[:, 2] - m[:, 0]) * (m[:, 3] - m[:, 1])).max()) if len(m) else np.nan for m in o.mtcnn]
    o['img_max'] = o[['pmc_FS', 'pmc_FA']].max(axis=1)
    o['noisy_or'] = 1 - np.prod(1 - o[[f'pmc_{d}' for d in DIMS]].values, axis=1)
    d_tab = o[o.grp.isin(['D', 'TN'])].groupby('grp').agg(
        n=('prob_fake', 'size'), 最大脸中位数=('maxface', 'median'), 图宽中位数=('W', 'median'),
        noisy_or中位数=('noisy_or', 'median'), 图像维最大值中位数=('img_max', 'median'))
    say(d_tab.round(4).to_string())

    open(f'{a.out_dir}/report.txt', 'w').write('\n'.join(out))
    print(f'\n[out] {a.out_dir}/report.txt')


if __name__ == '__main__':
    main(sys.argv[1:])
