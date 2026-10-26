"""B 阶段公共函数：冻结文件哈希校验、split 加载、来源字段、bootstrap。只用 CPU。"""
import os, sys, json, hashlib
import numpy as np
import pandas as pd

EXP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FROZEN = os.path.join(EXP_DIR, 'frozen_hypotheses.json')
FROZEN_SHA = os.path.join(EXP_DIR, 'results', 'frozen_hypotheses.sha256')
RESULTS = os.path.join(EXP_DIR, 'results')

FA_SCRIPTS = os.environ.get('FA_SCRIPTS', os.path.join(EXP_DIR, '../2026-10-07-asap-failure-analysis/scripts'))
sys.path.insert(0, FA_SCRIPTS)
from analyze_failures import DIMS, load_preds, build_frame  # noqa: E402,F401


def check_frozen(split):
    """返回冻结配置；与记录的 sha256 不一致时，test 直接退出，val 打印警告。"""
    sha = hashlib.sha256(open(FROZEN, 'rb').read()).hexdigest()
    rec = open(FROZEN_SHA).read().split()[0] if os.path.exists(FROZEN_SHA) else None
    if sha != rec:
        msg = f'frozen_hypotheses.json sha256={sha} 与记录 {rec} 不一致'
        if split == 'test':
            sys.exit('[abort] ' + msg)
        print('[warn] ' + msg)
    print(f'[frozen] sha256={sha}')
    return json.load(open(FROZEN)), sha


def load_split(cfg, split):
    """返回 (逐样本 DataFrame, metadata 列表)。DataFrame 含 pkl 字段、id/image/text/source/folder。"""
    inp = cfg['inputs']
    meta = json.load(open(os.path.join(inp['meta_dir'], f'{split}.json')))
    pkl = inp[f'pred_{split}']
    assert 'preUtilsC' not in pkl
    df = build_frame(load_preds(pkl, 0), meta)
    df['id'] = [meta[i]['id'] for i in df.meta_index]
    df['image'] = [meta[i]['image'] for i in df.meta_index]
    df['text'] = [meta[i]['text'] for i in df.meta_index]
    df['source'] = [source_of(p) for p in df.image]
    df['folder'] = df.image.str.rsplit('/', n=1).str[0]
    return df, meta


def source_of(path):
    """只对 /origin/ 路径取来源，其余记 unknown（冻结文件 source_field.rule）。"""
    return path.split('/')[2] if '/origin/' in path else 'unknown'


def cls6(fake_cls):
    return {'orig': 'orig', 'face_swap': 'FS', 'face_attribute': 'FA',
            'text_swap': 'TS', 'text_attribute': 'TA'}.get(fake_cls, 'mixed')


def boot_diff_props(a, b, B, rng):
    """两组 0/1（或比例）均值差 a-b 的 percentile bootstrap 95% CI，两组各自重采样。"""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) == 0 or len(b) == 0:
        return np.nan, np.nan
    d = [a[rng.integers(0, len(a), len(a))].mean() - b[rng.integers(0, len(b), len(b))].mean() for _ in range(B)]
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def out_dir(stage, split, smoke):
    d = os.path.join(RESULTS, 'smoke' if smoke else '', stage, split)
    os.makedirs(d, exist_ok=True)
    return d
