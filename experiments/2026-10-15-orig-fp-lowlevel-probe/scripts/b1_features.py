"""B1 单张图像特征（定义见 frozen_hypotheses.json 的 B1.feature_definitions / encoding_validity）。只用 CPU。"""
import os, sys, math
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from probe_lowlevel import immerkaer_sigma, lap_var, jpeg_quality, STD_LUMA  # noqa: E402

IMG_ROOT = os.environ.get('IMG_ROOT', '/nas/data_2/wanhongz/datasets')
QUALITY = ['lap_var', 'noise_sigma', 'noise_sigma_flat', 'jpeg_q', 'bytes_per_px', 'mean_saturation', 'near_gray',
           'bright_mean', 'bright_std', 'face_lap_var', 'face_noise_sigma', 'face_bg_ratio']
FACE_FEATS = ['face_lap_var', 'face_noise_sigma', 'face_bg_ratio']
ENCODING = ['jpeg_q', 'bytes_per_px']
FLAT_PCT = 20
MIN_CROP = 8


def immerkaer_response(g):
    return (g[:-2, :-2] - 2 * g[:-2, 1:-1] + g[:-2, 2:]
            - 2 * g[1:-1, :-2] + 4 * g[1:-1, 1:-1] - 2 * g[1:-1, 2:]
            + g[2:, :-2] - 2 * g[2:, 1:-1] + g[2:, 2:])


def flat_noise_sigma(g):
    """梯度幅值最低 20% 的内部像素上的 Immerkaer σ。"""
    if g.shape[0] < 3 or g.shape[1] < 3:
        return np.nan
    c = immerkaer_response(g)
    gx = (g[:-2, 2:] + 2 * g[1:-1, 2:] + g[2:, 2:]) - (g[:-2, :-2] + 2 * g[1:-1, :-2] + g[2:, :-2])
    gy = (g[2:, :-2] + 2 * g[2:, 1:-1] + g[2:, 2:]) - (g[:-2, :-2] + 2 * g[:-2, 1:-1] + g[:-2, 2:])
    mag = np.hypot(gx, gy)
    sel = mag <= np.percentile(mag, FLAT_PCT)
    return math.sqrt(math.pi / 2) * np.abs(c[sel]).mean() / 6


def jpeg_validity(im, q_est):
    """返回 jpeg_q 无效原因（'' 表示有效）。"""
    q = getattr(im, 'quantization', None)
    if not q or 0 not in q:
        return 'non_jpeg'
    if q_est >= 98:
        return 'q_ge_98'
    scale = 5000.0 / q_est if q_est < 50 else 200.0 - 2 * q_est
    pred = np.clip(np.floor((STD_LUMA * scale + 50) / 100), 1, 255)
    dev = np.abs(np.sort(np.asarray(q[0], dtype=np.float64)) - np.sort(pred)).mean() / pred.mean()
    return 'non_ijg' if dev > 0.15 else ''


def image_features(args):
    rel, mt = args
    path = os.path.join(IMG_ROOT, rel)
    try:
        size = os.path.getsize(path)
        im = Image.open(path)
        q_est = jpeg_quality(im)
        jinv = jpeg_validity(im, q_est)
        meta_bytes = len(im.info.get('exif', b'') or b'') + len(im.info.get('icc_profile', b'') or b'')
        mode = im.mode
        rgb = im.convert('RGB')
        g = np.asarray(rgb.convert('L'), dtype=np.float64)
        Hi, Wi = g.shape
        sat = float(np.asarray(rgb.convert('HSV'))[:, :, 1].mean() / 255)
        lap = g[:-2, 1:-1] + g[2:, 1:-1] + g[1:-1, :-2] + g[1:-1, 2:] - 4 * g[1:-1, 1:-1]
        r = dict(image=rel, err='', img_w=Wi, img_h=Hi, mode=mode,
                 lap_var=float(lap.var()), noise_sigma=immerkaer_sigma(g), noise_sigma_flat=flat_noise_sigma(g),
                 jpeg_q_raw=q_est, jpeg_q_invalid=jinv, jpeg_q=np.nan if jinv else q_est,
                 meta_bytes_frac=meta_bytes / size, bytes_per_px_invalid=int(meta_bytes > 0.1 * size),
                 mean_saturation=sat, near_gray=int(mode in ('L', 'LA', '1') or sat < 0.05),
                 bright_mean=float(g.mean()), bright_std=float(g.std()),
                 face_lap_var=np.nan, face_noise_sigma=np.nan, face_bg_ratio=np.nan, face_small=0)
        r['bytes_per_px'] = np.nan if r['bytes_per_px_invalid'] else size / (Wi * Hi)
        if len(mt):
            areas = (mt[:, 2] - mt[:, 0]) * (mt[:, 3] - mt[:, 1])
            x1, y1, x2, y2 = mt[int(np.argmax(areas))]
            x1, x2 = int(round(max(x1, 0))), int(round(min(x2, Wi)))
            y1, y2 = int(round(max(y1, 0))), int(round(min(y2, Hi)))
            if x2 - x1 < MIN_CROP or y2 - y1 < MIN_CROP:
                r['face_small'] = 1
            else:
                crop = g[y1:y2, x1:x2]
                r['face_lap_var'] = lap_var(crop)
                r['face_noise_sigma'] = immerkaer_sigma(crop)
                mask = np.ones_like(g, dtype=bool)
                for b in mt:
                    bx1, bx2 = int(round(max(b[0], 0))), int(round(min(b[2], Wi)))
                    by1, by2 = int(round(max(b[1], 0))), int(round(min(b[3], Hi)))
                    mask[by1:by2, bx1:bx2] = False
                bg = lap[mask[1:-1, 1:-1]]
                if bg.size >= 100 and bg.var() > 0 and r['face_lap_var'] > 0:
                    r['face_bg_ratio'] = math.log(r['face_lap_var'] / bg.var())
        return r
    except Exception as e:  # noqa: BLE001
        return dict(image=rel, err=repr(e)[:200])
