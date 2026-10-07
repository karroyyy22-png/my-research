# export_preds.py —— 逐样本导出纯 ASAP（不含 OmniFD）的预测结果，用于 failure mode 分析
import os, sys, argparse, pickle, runpy, importlib.util, time
os.environ["TOKENIZERS_PARALLELISM"] = "false"
HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE); sys.path.insert(0, HERE)

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import ruamel.yaml as yaml
from PIL import Image
from torchvision import transforms
from transformers import BertTokenizerFast

from models.vit import interpolate_pos_embed
from models import box_ops
from models.ASAP import ASAP
from dataset.dataset import DGM4_Dataset
from tools.multilabel_metrics import get_multi_label

BERT_DIR = '/nas/data_2/wanhongz/bert-base-uncased'
META_DIR = '/nas/data_2/wanhongz/datasets/DGM4/metadata'
LOG_DIR = '/nas/data_2/wanhongz/ASAP/results/log20260602_134300_train_4gpu_batch16'


def load_test_module(here):
    spec = importlib.util.spec_from_file_location("asap_test", os.path.join(here, "test.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def get_default_test_args(here):
    """把 test.py 的 __main__ 运行到 parse_args 为止，取出其全部默认参数。"""
    class _Got(Exception):
        pass
    orig = argparse.ArgumentParser.parse_args

    def grab(self, args=None, namespace=None):
        raise _Got(orig(self, args=[]))
    argparse.ArgumentParser.parse_args = grab
    try:
        runpy.run_path(os.path.join(here, "test.py"), run_name="__main__")
    except _Got as e:
        return e.args[0]
    finally:
        argparse.ArgumentParser.parse_args = orig
    raise RuntimeError("test.py 的 __main__ 没有调用 parse_args")


def load_config(path):
    try:
        return yaml.load(open(path, 'r'), Loader=yaml.Loader)
    except Exception:
        return yaml.YAML(typ='safe', pure=True).load(open(path, 'r'))


def build_loader(config, split, n, batch_size, num_workers):
    normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                                     (0.26862954, 0.26130258, 0.27577711))
    tf = transforms.Compose([
        transforms.Resize((config['image_res'], config['image_res']), interpolation=Image.BICUBIC),
        transforms.ToTensor(),
        normalize,
    ])
    ds = DGM4_Dataset(config=config, ann_file=[f'{META_DIR}/{split}.json'], transform=tf,
                      max_words=config['max_words'], is_train=False)
    idx = list(range(len(ds))) if n <= 0 else list(range(min(n, len(ds))))
    loader = DataLoader(Subset(ds, idx), batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=True, drop_last=False)
    return ds, loader, idx


def build_model(model_args, config, ckpt_path, device):
    tokenizer = BertTokenizerFast.from_pretrained(BERT_DIR)
    model = ASAP(args=model_args, config=config, text_encoder=BERT_DIR, tokenizer=tokenizer, init_deit=True)
    model = model.to(device)
    sd = torch.load(ckpt_path, map_location='cpu')['model']
    sd['visual_encoder.pos_embed'] = interpolate_pos_embed(sd['visual_encoder.pos_embed'], model.visual_encoder)
    msg = model.load_state_dict(sd, strict=False)
    print(f"[load] missing_keys={len(msg.missing_keys)}  unexpected_keys={len(msg.unexpected_keys)}")
    for k in msg.missing_keys[:30]:
        print("   missing   :", k)
    for k in msg.unexpected_keys[:10]:
        print("   unexpected:", k)
    model.eval()
    return model, tokenizer


def to_np(x):
    return x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)


@torch.no_grad()
def run_inference(model, tokenizer, loader, idx_list, ann, test_mod, device):
    rows, ptr, t0 = [], 0, time.time()
    for it, (image, label, text, fake_image_box, fake_word_pos, W, H, real_cap, real_prom,
             res_fake_pos, res_fake_pos_patch, img_path) in enumerate(loader):
        B = image.size(0)
        image = image.to(device, non_blocking=True)
        kw = dict(max_length=128, truncation=True, add_special_tokens=True,
                  return_attention_mask=True, return_token_type_ids=False)
        text_input = tokenizer(text, **kw)
        cap_input = tokenizer(real_cap, **kw)
        prom_input = tokenizer(real_prom, **kw)
        text_input, fake_token_pos, subword_idx = test_mod.text_input_adjust(text_input, fake_word_pos, device)
        cap_input = test_mod.text_input_adjust(cap_input, fake_word_pos, device, True)
        prom_input = test_mod.text_input_adjust(prom_input, fake_word_pos, device, True)

        logits_rf, logits_mc, coord, logits_tok = model(
            image, label, text_input, fake_image_box, fake_token_pos, cap_input, prom_input,
            res_fake_pos, res_fake_pos_patch, is_train=False, omni_feat=None)

        prob_fake = to_np(F.softmax(logits_rf, dim=1)[:, 1])
        pred_cls = to_np(logits_rf.argmax(1))
        prob_mc = to_np(torch.sigmoid(logits_mc))
        target_mc, _ = get_multi_label(label, image)
        target_mc = to_np(target_mc)

        boxes1 = box_ops.box_cxcywh_to_xyxy(coord)
        boxes2 = box_ops.box_cxcywh_to_xyxy(fake_image_box.to(device))
        iou, _ = box_ops.box_iou(boxes1, boxes2, test=True)
        iou = to_np(iou).reshape(-1)
        coord_np = to_np(coord)

        token_label = text_input.attention_mask[:, 1:].clone()
        token_label[token_label == 0] = -100
        token_label[token_label == 1] = 0
        for b, pos in enumerate(fake_token_pos):
            for p in pos:
                token_label[b, p] = 1
        logits_tok = logits_tok.view(B, -1, 2)
        tok_prob_all = F.softmax(logits_tok, dim=-1)[..., 1]
        tok_pred_all = logits_tok.argmax(-1)

        for b in range(B):
            mi = idx_list[ptr + b]
            assert img_path[b] == ann[mi]['image'], f"顺序错位: {mi} {img_path[b]} vs {ann[mi]['image']}"
            valid = token_label[b] != -100
            tl = to_np(token_label[b][valid]).astype(int)
            tp_ = to_np(tok_pred_all[b][valid]).astype(int)
            tpr = to_np(tok_prob_all[b][valid]).astype(np.float32)
            wid = np.array([-1 if w is None else w for w in subword_idx[b]], dtype=int)
            assert len(wid) == len(tl), f"token 长度不一致: {mi} {len(wid)} vs {len(tl)}"
            Wb, Hb = int(W[b]), int(H[b])
            cx, cy, bw, bh = coord_np[b].tolist()
            rows.append(dict(
                meta_index=mi, image=img_path[b], fake_cls=label[b], W=Wb, H=Hb,
                prob_fake=float(prob_fake[b]), pred_cls=int(pred_cls[b]),
                prob_mc=prob_mc[b].astype(np.float32), target_mc=target_mc[b].astype(np.int8),
                pred_box_norm_cxcywh=np.array([cx, cy, bw, bh], dtype=np.float32),
                pred_box_px_xyxy=np.array([(cx - bw / 2) * Wb, (cy - bh / 2) * Hb,
                                           (cx + bw / 2) * Wb, (cy + bh / 2) * Hb], dtype=np.float32),
                iou_official=float(iou[b]),
                tok_label=tl.astype(np.int8), tok_pred=tp_.astype(np.int8), tok_prob=tpr, tok_word_id=wid,
                tok_tp=int(((tl == 1) & (tp_ == 1)).sum()), tok_fp=int(((tl == 0) & (tp_ == 1)).sum()),
                tok_fn=int(((tl == 1) & (tp_ == 0)).sum()), tok_tn=int(((tl == 0) & (tp_ == 0)).sum()),
                pred_fake_words=sorted(set(wid[tp_ == 1].tolist()) - {-1}),
                gt_fake_words=np.where(to_np(fake_word_pos[b]) == 1)[0].tolist(),
            ))
        ptr += B
        if it % 50 == 0:
            print(f"  batch {it}/{len(loader)}  samples={ptr}  {time.time() - t0:.0f}s", flush=True)
    return rows


def check_against_npz(rows, npz_path):
    z = np.load(npz_path, allow_pickle=True)
    mi = np.array([r['meta_index'] for r in rows])
    ours = np.array([r['prob_fake'] for r in rows])
    ref = z['y_pred_asap'][mi]
    path_ok = all(r['image'] == p for r, p in zip(rows, z['img_path_all'][mi]))
    d = np.abs(ours - ref)
    print(f"[check vs 7/21 npz] 路径一致={path_ok}  |Δprob| max={d.max():.2e} mean={d.mean():.2e}  "
          f"判定(0.5)一致率={np.mean((ours >= .5) == (ref >= .5)):.4f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--split', default='test', choices=['test', 'val'])
    p.add_argument('--n', type=int, default=200, help='<=0 表示全量')
    p.add_argument('--bs', type=int, default=64)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--ckpt', default=f'{LOG_DIR}/checkpoint_best.pth')
    p.add_argument('--config', default=f'{LOG_DIR}/config.yaml')
    p.add_argument('--ref_npz', default=f'{LOG_DIR}/evaluation/raw_eval_data.npz')
    p.add_argument('--out_dir', default='/nas/data_2/wanhongz/failure_analysis')
    a = p.parse_args()

    test_mod = load_test_module(HERE)
    model_args = get_default_test_args(HERE)
    model_args.distributed, model_args.use_omni_fusion, model_args.log = False, False, False
    model_args.gpu, model_args.device = 0, 'cuda'

    torch.manual_seed(777); np.random.seed(777)
    device = torch.device('cuda')
    config = load_config(a.config)

    print("[multi-label 顺序]", to_np(get_multi_label(
        ['face_swap', 'face_attribute', 'text_swap', 'text_attribute', 'orig'], torch.zeros(5, 1))[0]).tolist())

    model, tokenizer = build_model(model_args, config, a.ckpt, device)
    ds, loader, idx = build_loader(config, a.split, a.n, a.bs, a.workers)
    print(f"[data] split={a.split}  总数={len(ds)}  本次={len(idx)}")

    rows = run_inference(model, tokenizer, loader, idx, ds.ann, test_mod, device)

    tag = 'all' if a.n <= 0 else f'n{a.n}'
    out = os.path.join(a.out_dir, f'preds_{a.split}_{tag}.pkl')
    with open(out, 'wb') as f:
        pickle.dump(rows, f)
    print(f"[save] {out}  rows={len(rows)}")

    if a.split == 'test' and os.path.exists(a.ref_npz):
        check_against_npz(rows, a.ref_npz)
    r = rows[0]
    print("[样例]", {k: r[k] for k in ['meta_index', 'fake_cls', 'W', 'H', 'prob_fake', 'iou_official',
                                      'pred_fake_words', 'gt_fake_words']})


if __name__ == '__main__':
    main()
