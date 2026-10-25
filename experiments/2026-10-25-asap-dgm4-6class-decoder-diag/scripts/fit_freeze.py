"""只在 val_dev 上拟合所有阈值/偏置/校准器；在 val_dev 与 val_holdout 上报告；把配置冻结到 results/frozen_config.json。

规则选择（val_holdout 确认）：调参版本相对未调参版本的 Macro-F1(6) 差值在 val_holdout 上做 paired bootstrap，
记录在 frozen_config.json 的 holdout_confirmation 中。test 评测不改动任何参数，所有解码器都在 test 上报告。
"""
import os, json, hashlib
import numpy as np
from common import *
from pipeline import *
from pipeline import _A_states

path = f'{RES_DIR}/frozen_config.json'
assert not os.path.exists(path), f'{path} 已存在；冻结后不再重新拟合'
split = json.load(open(f'{RES_DIR}/split_val.json'))
val = load_frame('val').set_index('meta_index', drop=False)
dev, hold = val.loc[split['val_dev']].reset_index(drop=True), val.loc[split['val_holdout']].reset_index(drop=True)
assert len(set(split['val_dev']) & set(split['val_holdout'])) == 0 and len(dev) + len(hold) == len(val)

# 网格搜索用的向量化 A 与 common.decode_A 必须完全一致
pf, P = probs(dev)
for tb, tm in [(0.5, 0.5), (0.3, 0.7), (0.8, 0.2)]:
    assert np.array_equal(decode_A(pf, P, tb, tm), np.where(pf < tb, 0, _A_states(P, tm)))

cfg = fit_all(dev)
for k, v in cfg.items():
    print(k, {a: b for a, b in v.items() if a not in ('coef', 'intercept', 'standardize_mu', 'standardize_sd')})

summ = {}
for name, df in [('val_dev', dev), ('val_holdout', hold)]:
    res = report_split(cfg, df, name, do_e02=(name == 'val_holdout'))
    summ[name] = res
    print(f'\n== {name} ==')
    print(res['e01']['main'][['decoder', 'macroF1_6', 'balAcc_6', 'acc_6', 'orig_R']].round(4).to_string(index=False))

boot = summ['val_holdout']['e01']['boot']
conf = {}
for a, c in [('A_tuned', 'A_untuned'), ('B_tuned', 'B_b0'), ('B_TS_tuned', 'B_tuned')]:
    r = boot[(boot.pair == f'{a} - {c}') & (boot.metric == 'macroF1_6')].iloc[0]
    verdict = '支持' if r.ci95_lo > 0 else ('不支持' if r.ci95_hi < 0 else '证据不足')
    conf[f'{a} vs {c}'] = dict(diff=float(r['diff']), ci95=[float(r.ci95_lo), float(r.ci95_hi)],
                               claim=f'{a} 的六类 Macro-F1 高于 {c}', verdict=verdict)
main_h = summ['val_holdout']['e01']['main'].set_index('decoder')
primary = str(main_h.loc[DECODERS, 'macroF1_6'].idxmax())
frozen = dict(
    note='所有参数只在 val_dev 上拟合；val_holdout 只用于确认；test 只评估一次，不据 test 结果修改任何规则。',
    data=dict(preds_val=PRED_PATH['val'], preds_test=PRED_PATH['test'], split_file='results/split_val.json',
              n_val_dev=len(dev), n_val_holdout=len(hold)),
    search=dict(A_grid_t_b=[float(GRID_T[0]), float(GRID_T[-1]), 0.01], A_grid_t_m=[float(GRID_T[0]), float(GRID_T[-1]), 0.01],
                B_grid_b_orig=[float(GRID_B[0]), float(GRID_B[-1]), 0.05], objective='val_dev 六类 Macro-F1',
                tie_break='A: 距 (0.5,0.5) 最近；B: |b_orig| 最小', clip=EPS,
                match_rule='A_matchB: t_m 固定为 A_tuned 的 t_m，t_b 网格步长 0.001，使 val_dev Original Recall 最接近 B_tuned；'
                           'B_matchA: b_orig 网格同 B，使 val_dev Original Recall 最接近 A_tuned',
                temperature='四维各自一个温度 T，在 val_dev 上最小化 BCE（logit/T），有界搜索 log T∈[-3,3]',
                bootstrap=dict(n=N_BOOT, seed=BOOT_SEED, method='multinomial 重采样权重，配对')),
    decoders=cfg,
    holdout_confirmation=conf,
    primary_decoder_by_holdout_macroF1_6=primary,
    e02_decoders=E02_DECODERS,
)
frozen['sha256_of_decoders'] = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()
json.dump(frozen, open(path, 'w'), indent=2, ensure_ascii=False)
print('\nholdout 确认:', json.dumps(conf, ensure_ascii=False, indent=1))
print('primary:', primary, '\n已写入', path)
