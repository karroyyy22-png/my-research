import re, sys
P = '/nas/data_2/wanhongz/ASAP/test.py'
src = open(P, encoding='utf-8').read()
if '[phase0]' in src:
    sys.exit('[ABORT] test.py 里已经有 [phase0] 修改，不重复打补丁')

def _one(pat, s, name):
    ms = list(re.finditer(pat, s))
    if len(ms) != 1:
        sys.exit(f'[ABORT] {name}: 匹配 {len(ms)} 次（要求 1 次），未写入任何修改')
    return ms[0]

def insert_after(s, literal, text, name):
    m = _one(re.escape(literal) + r'[ \t]*\n', s, name)
    return s[:m.end()] + text + s[m.end():]

def insert_before_line(s, literal, text, name):
    m = _one(r'(?m)^[ \t]*' + re.escape(literal), s, name)
    return s[:m.start()] + text + s[m.start():]

ARGS = """    parser.add_argument('--box_postproc', default='none', choices=['none', 'cscl', 'zero'],
                        help='[phase0] 框后处理: none=原样; cscl=FS/FA logit都<0.5时置零(同CSCL); zero=logit都<0时置零')
    parser.add_argument('--max_samples', default=0, type=int, help='[phase0] 只评测前约N个样本(冒烟测试), 0=全部')
    parser.add_argument('--eval_dirname', default='evaluation', type=str, help='[phase0] 评测结果保存的子目录名')
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

POSTPROC = """        # ===== [phase0] 框后处理（默认 none = 原始行为） =====
        if args.box_postproc != 'none':
            _thr = 0.5 if args.box_postproc == 'cscl' else 0.0
            _pp_mask = (logits_multicls[:, 0] < _thr) & (logits_multicls[:, 1] < _thr)
            output_coord = output_coord.clone()
            output_coord[_pp_mask] = 0
"""

STATS = """        # ===== [phase0] 拆分统计 =====
        _pred_xyxy = boxes1.detach().float().cpu()
        _gt_xyxy = boxes2.detach().float().cpu()
        _gt_empty_b = (fake_image_box.detach().float().cpu().abs().sum(dim=1) == 0)
        _pred_empty_b = (_pred_xyxy < 1e-4).all(dim=1)
        _ph0_mc.append(logits_multicls[:, :2].detach().float().cpu())
        _Wl = W.tolist() if torch.is_tensor(W) else list(W)
        _Hl = H.tolist() if torch.is_tensor(H) else list(H)
        for _k in range(len(img_path)):
            _is_e = bool(_gt_empty_b[_k])
            _ph0_gt_empty.append(_is_e)
            _ph0_pred_empty.append(bool(_pred_empty_b[_k]))
            if _is_e:
                _ph0_cand.append(-1)
                continue
            _m = _ph0_meta.get(img_path[_k])
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
    def _ph0_mean(a):
        return float(np.mean(a) * 100) if len(a) > 0 else float('nan')
    _gem = np.array(_ph0_gt_empty, dtype=bool)
    _pem = np.array(_ph0_pred_empty, dtype=bool)
    _cdm = np.array(_ph0_cand)
    _iou_t = IOU_pred_all_np[~_gem]
    _iou_u = IOU_pred_all_np[_gem]
    _mc = torch.cat(_ph0_mc, 0).numpy() if len(_ph0_mc) > 0 else np.zeros((0, 2))
    _ph0 = {
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

s = src
s = insert_after(s, "parser.add_argument('--test_epoch', default='best', type=str)", ARGS, 'ARGS')
s = insert_after(s, "    IOU_pred_all = []", INIT, 'INIT')
s = insert_after(s, "in enumerate(metric_logger.log_every(args, data_loader, print_freq, header)):", LOOPSTART, 'LOOPSTART')
s = insert_before_line(s, "boxes1 = box_ops.box_cxcywh_to_xyxy(output_coord)", POSTPROC, 'POSTPROC')
s = insert_after(s, "IOU_pred_all.extend(IOU.cpu().tolist())", STATS, 'STATS')
s = insert_after(s, "IOU_ACC_95 = np.mean(IOU_pred_all_np > 0.95)", SUMMARY, 'SUMMARY')

old = "os.path.join(args.output_dir, args.log_num, 'evaluation'"
n = s.count(old)
if n != 3:
    sys.exit(f"[ABORT] 'evaluation' 路径匹配 {n} 次（预期 3 次），未写入任何修改")
s = s.replace(old, "os.path.join(args.output_dir, args.log_num, args.eval_dirname")

open(P, 'w', encoding='utf-8').write(s)
print('[OK] 补丁已写入')
