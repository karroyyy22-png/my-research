import re, sys
T = 'test.py'
M = 'models/CSCL.py'
ts = open(T, encoding='utf-8').read()
ms = open(M, encoding='utf-8').read()
if '[phase0]' in ts or '[phase0]' in ms:
    sys.exit('[ABORT] 已经有 [phase0] 修改，不重复打补丁')

def _one(pat, s, name):
    found = list(re.finditer(pat, s))
    if len(found) != 1:
        sys.exit(f'[ABORT] {name}: 匹配 {len(found)} 次（要求 1 次），未写入任何修改')
    return found[0]

def insert_after(s, literal, text, name):
    m = _one(re.escape(literal) + r'[ \t]*\n', s, name)
    return s[:m.end()] + text + s[m.end():]

def insert_after_line_re(s, pat, text, name):
    m = _one(pat, s, name)
    return s[:m.end()] + text + s[m.end():]

# ---------- CSCL.py：清零改成可切换，默认 cscl = 原始行为 ----------
pp_pat = (r'(?m)^[ \t]*' + re.escape('logits_multicls_mask = ((logits_multicls[:, 0]<0.5)&(logits_multicls[:, 1]<0.5))')
          + r'[ \t]*\n[ \t]*' + re.escape('output_coord[logits_multicls_mask] = torch.tensor([0.0, 0.0, 0.0, 0.0]).to(output_coord.device)')
          + r'[ \t]*\n')
m = _one(pp_pat, ms, 'CSCL.py 后处理')
PP = """            # ===== [phase0] 后处理开关：默认 cscl = 原始行为 =====
            _pp = getattr(self, 'phase0_box_postproc', 'cscl')
            if _pp != 'none':
                _thr = 0.5 if _pp == 'cscl' else 0.0
                logits_multicls_mask = ((logits_multicls[:, 0]<_thr)&(logits_multicls[:, 1]<_thr))
                output_coord[logits_multicls_mask] = torch.tensor([0.0, 0.0, 0.0, 0.0]).to(output_coord.device)
"""
ms = ms[:m.start()] + PP + ms[m.end():]

# ---------- test.py ----------
ARGS = """    parser.add_argument('--box_postproc', default='cscl', choices=['none', 'cscl', 'zero'],
                        help='[phase0] 框后处理: cscl=原始行为(FS/FA logit都<0.5时置零); none=不置零; zero=logit都<0时置零')
    parser.add_argument('--max_samples', default=0, type=int, help='[phase0] 只评测前约N个样本(冒烟测试), 0=全部')
    parser.add_argument('--eval_dirname', default='evaluation', type=str, help='[phase0] 评测结果保存的子目录名')
"""
SETPP = """    _ph0_model = model.module if hasattr(model, 'module') else model  # [phase0]
    _ph0_model.phase0_box_postproc = args.box_postproc
    print(f'[phase0] box_postproc = {args.box_postproc}')
"""
INIT = """    # ===== [phase0] 拆分统计用的变量 =====
    import json as _ph0_json
    _ph0_vf = config['val_file']
    _ph0_vf = _ph0_vf[0] if isinstance(_ph0_vf, (list, tuple)) else _ph0_vf
    _ph0_meta = {}
    with open(_ph0_vf, encoding='utf-8') as _f:
        for _r in _ph0_json.load(_f):
            _ph0_meta[_r['image']] = {'gt': list(_r.get('fake_image_box') or []),
                                      'mtcnn': [list(b) for b in (_r.get('mtcnn_boxes') or [])]}
    _ph0_gt_empty, _ph0_pred_empty, _ph0_cand, _ph0_mc = [], [], [], []
    _ph0_gt_mismatch = 0
    _ph0_seen = 0

    def _ph0_key(p):
        return p.split('datasets/', 1)[1] if 'datasets/' in p else p

    def _ph0_iou_1xn(b, bs):
        lt = torch.max(b[:2], bs[:, :2])
        rb = torch.min(b[2:], bs[:, 2:])
        wh = (rb - lt).clamp(min=0)
        inter = wh[:, 0] * wh[:, 1]
        a1 = (b[2] - b[0]).clamp(min=0) * (b[3] - b[1]).clamp(min=0)
        a2 = (bs[:, 2] - bs[:, 0]).clamp(min=0) * (bs[:, 3] - bs[:, 1]).clamp(min=0)
        return inter / (a1 + a2 - inter + 1e-12)
"""
LOOPSTART = """        if args.max_samples > 0 and _ph0_seen >= args.max_samples:  # [phase0]
            break
        _ph0_seen += len(label)
"""
SNAP = """        _ph0_mc_raw = logits_multicls[:, :2].detach().float().cpu().clone()  # [phase0] 在下面被原地二值化之前保存原始 logit
"""
STATS = """        # ===== [phase0] 拆分统计 =====
        _pred_xyxy = boxes1.detach().float().cpu()
        _gt_xyxy = boxes2.detach().float().cpu()
        _gt_empty_b = (fake_image_box.detach().float().cpu().abs().sum(dim=1) == 0)
        _pred_empty_b = (_pred_xyxy < 1e-4).all(dim=1)
        _ph0_mc.append(_ph0_mc_raw)
        _Wl = W.tolist() if torch.is_tensor(W) else list(W)
        _Hl = H.tolist() if torch.is_tensor(H) else list(H)
        for _k in range(len(image_path)):
            _is_e = bool(_gt_empty_b[_k])
            _ph0_gt_empty.append(_is_e)
            _ph0_pred_empty.append(bool(_pred_empty_b[_k]))
            if _is_e:
                _ph0_cand.append(-1)
                continue
            _m = _ph0_meta.get(_ph0_key(image_path[_k]))
            if _m is None or len(_m['mtcnn']) == 0 or len(_m['gt']) != 4:
                _ph0_cand.append(-2)
                continue
            _scale = torch.tensor([_Wl[_k], _Hl[_k], _Wl[_k], _Hl[_k]], dtype=torch.float)
            _gt_norm = torch.tensor(_m['gt'], dtype=torch.float) / _scale
            if float((_gt_norm - _gt_xyxy[_k]).abs().max()) > 1e-3:
                _ph0_gt_mismatch += 1
            _ious = _ph0_iou_1xn(_pred_xyxy[_k], torch.tensor(_m['mtcnn'], dtype=torch.float) / _scale)
            _j = int(torch.argmax(_ious))
            _ok = float(_ious[_j]) > 0 and _m['mtcnn'][_j] == _m['gt']
            _ph0_cand.append(1 if _ok else 0)
"""
SUMMARY = """    # ===== [phase0] 拆分指标汇总 =====
    IOU_pred_all_np = np.array(IOU_pred, dtype=float)
    def _ph0_mean(a):
        return float(np.mean(a) * 100) if len(a) > 0 else float('nan')
    _gem = np.array(_ph0_gt_empty, dtype=bool)
    _pem = np.array(_ph0_pred_empty, dtype=bool)
    _cdm = np.array(_ph0_cand)
    _iou_t = IOU_pred_all_np[~_gem]
    _iou_u = IOU_pred_all_np[_gem]
    _mc = torch.cat(_ph0_mc, 0).numpy() if len(_ph0_mc) > 0 else np.zeros((0, 2))
    _ph0 = {
        'model': 'CSCL',
        'box_postproc': args.box_postproc,
        'checkpoint': f'{args.log_num}/checkpoint_{args.test_epoch}.pth',
        'n_all': int(len(IOU_pred_all_np)),
        'n_tampered': int((~_gem).sum()),
        'n_untampered': int(_gem.sum()),
        'all_IoUmean': _ph0_mean(IOU_pred_all_np),
        'all_IoU50': _ph0_mean(IOU_pred_all_np > 0.5),
        'all_IoU75': _ph0_mean(IOU_pred_all_np > 0.75),
        'tampered_IoUmean': _ph0_mean(_iou_t),
        'tampered_IoU50': _ph0_mean(_iou_t > 0.5),
        'tampered_IoU75': _ph0_mean(_iou_t > 0.75),
        'tampered_pred_empty_rate': _ph0_mean(_pem[~_gem]),
        'untampered_pred_empty_rate': _ph0_mean(_pem[_gem]),
        'untampered_IoUmean': _ph0_mean(_iou_u),
        'cand_acc_tampered': _ph0_mean(_cdm[~_gem] == 1),
        'cand_meta_missing': int((_cdm == -2).sum()),
        'gt_norm_mismatch': int(_ph0_gt_mismatch),
        'img_logit_min': float(_mc.min()) if _mc.size else float('nan'),
        'img_logit_max': float(_mc.max()) if _mc.size else float('nan'),
    }
    print('[phase0] ' + _ph0_json.dumps(_ph0, ensure_ascii=False, indent=2))
    _ph0_dir = os.path.join(args.output_dir, args.log_num, args.eval_dirname)
    os.makedirs(_ph0_dir, exist_ok=True)
    with open(os.path.join(_ph0_dir, f'phase0_split_{args.box_postproc}.json'), 'w', encoding='utf-8') as _f:
        _ph0_json.dump(_ph0, _f, ensure_ascii=False, indent=2)
"""

ts = insert_after(ts, "parser.add_argument('--test_epoch', default='49', type=str)", ARGS, 'ARGS')
ts = insert_after_line_re(ts, r'(?m)^    model\.eval\(\)[ \t]*\n', SETPP, 'SETPP')
ts = insert_after(ts, "    y_true, y_pred, IOU_pred, IOU_50, IOU_75, IOU_95 = [], [], [], [], [], []", INIT, 'INIT')
ts = insert_after(ts, "in enumerate(metric_logger.log_every(args, data_loader, print_freq, header)):", LOOPSTART, 'LOOPSTART')
ts = insert_after(ts, "logits_real_fake, logits_multicls, output_coord, logits_tok = model(image, label, text_input, fake_image_box, fake_token_pos, is_train=False)", SNAP, 'SNAP')
ts = insert_after(ts, "IOU_pred.extend(IOU.cpu().tolist())", STATS, 'STATS')
ts = insert_after(ts, "IOU_ACC_75 = sum(IOU_75)/len(IOU_75)", SUMMARY, 'SUMMARY')

old = "os.path.join(args.output_dir, args.log_num, 'evaluation'"
n = ts.count(old)
if n != 1:
    sys.exit(f"[ABORT] 'evaluation' 路径匹配 {n} 次（预期 1 次），未写入任何修改")
ts = ts.replace(old, "os.path.join(args.output_dir, args.log_num, args.eval_dirname")

open(M, 'w', encoding='utf-8').write(ms)
open(T, 'w', encoding='utf-8').write(ts)
print('[OK] CSCL.py 和 test.py 补丁已写入')
