# -*- coding: utf-8 -*-
"""
评估: 用 OmniFD 做候选框选择，能达到多少 IoU
任务性质: 每张伪造图有 2~5 个 mtcnn 候选框，GT 恰好等于其中一个 (IoU=1.0)
对比基线:
  - 随机选           ~33%
  - 总是选第一张      66.21% (train) / 59~75% (test 分类别)
  - ASAP 当前实测     74.43%
  - 上限             100%
"""
import sys, os, json, numpy as np, torch
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from collections import defaultdict

sys.path.insert(0, '/nas/data_2/wanhongz/code/OmniFD')
from mmengine.config import Config
from mmpretrain.registry import MODELS

ANN   = '/nas/data_2/wanhongz/datasets/DGM4/metadata/test.json'
ROOT  = '/nas/data_2/wanhongz/datasets/'
CKPT  = '/nas/data_2/wanhongz/code/OmniFD/pretrained/omnifd_swin-s_50k_forgerynet.pth'
CFG   = '/nas/data_2/wanhongz/code/OmniFD/configs_omnifd/single/omnifd_swin-s_50k_forgerynet_image.py'
OUT   = '/nas/data_2/wanhongz/face_cropped_features/boxselect_eval.npz'
MARGIN = 0.0          # 裁剪外扩比例，0 表示紧贴框（跟原提取脚本一致）

# ---------- 展开成 (样本, 候选框) 的扁平列表 ----------
data = json.load(open(ANN))
items = []
for si, it in enumerate(data):
    if it['fake_cls'] == 'orig':
        continue
    gt, mb = it.get('fake_image_box'), it.get('mtcnn_boxes')
    if not gt or not mb:
        continue
    for bi, b in enumerate(mb):
        items.append((si, bi, it['image'], b))
print(f'伪造样本(有框): {len(set(i[0] for i in items))}, 候选框总数: {len(items)}')

# ---------- 模型 ----------
print('加载 OmniFD...')
cfg = Config.fromfile(CFG)
# data_preprocessor 定义在配置顶层而不是 cfg.model 里，
# 而 UnifiedDetector.__init__ 会无条件 MODELS.build 它（空 dict 会报错），
# 这里显式补上。本脚本自己做 resize/normalize，不依赖它的实际行为。
if not cfg.model.get('data_preprocessor'):
    cfg.model['data_preprocessor'] = cfg.get(
        'data_preprocessor', dict(type='UnifiedDataPreprocessor'))
model = MODELS.build(cfg.model)
ck = torch.load(CKPT, map_location='cpu')
msg = model.load_state_dict(ck.get('state_dict', ck), strict=False)
print(f'  missing={len(msg.missing_keys)}  unexpected={len(msg.unexpected_keys)}')
if msg.missing_keys[:5]:
    print('  missing 前5:', msg.missing_keys[:5])
model.eval().cuda()
image_head = model.head.decoder_modules['image']

_orig_fwd = model.neck.forward
def _patched(inputs, data_samples=None, **kw):
    class FakeDS: data_type = 'image'
    return _orig_fwd(inputs, data_samples=[FakeDS()], **kw)
model.neck.forward = _patched

tf = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])

class BoxDS(Dataset):
    def __len__(self): return len(items)
    def __getitem__(self, i):
        si, bi, path, box = items[i]
        try:
            img = Image.open(os.path.join(ROOT, path)).convert('RGB')
            x1, y1, x2, y2 = box
            if MARGIN > 0:
                w, h = x2 - x1, y2 - y1
                x1 -= w*MARGIN; x2 += w*MARGIN; y1 -= h*MARGIN; y2 += h*MARGIN
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(img.width, x2), min(img.height, y2)
            crop = img.crop((x1, y1, x2, y2))
        except Exception:
            crop = Image.new('RGB', (224, 224))
        return tf(crop), si, bi

loader = DataLoader(BoxDS(), batch_size=64, num_workers=8, shuffle=False)
mean = torch.tensor([.485,.456,.406]).view(1,3,1,1).cuda()
std  = torch.tensor([.229,.224,.225]).view(1,3,1,1).cuda()

scores = np.zeros(len(items), dtype=np.float32)
p = 0
with torch.no_grad():
    for imgs, si, bi in tqdm(loader, desc='OmniFD 打分'):
        imgs = (imgs.cuda(non_blocking=True) - mean) / std
        logit = image_head(model.neck(model.backbone(imgs)))
        s = torch.softmax(logit, -1)[:, 1].cpu().numpy()
        scores[p:p+len(s)] = s; p += len(s)

# ---------- 按样本聚合、选框、算 IoU ----------
def iou(a, b):
    x1,y1,x2,y2 = max(a[0],b[0]), max(a[1],b[1]), min(a[2],b[2]), min(a[3],b[3])
    if x2<=x1 or y2<=y1: return 0.0
    i = (x2-x1)*(y2-y1)
    u = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - i
    return i/u if u>0 else 0.0

per = defaultdict(list)
for k, (si, bi, _, box) in enumerate(items):
    per[si].append((bi, box, scores[k]))

res = defaultdict(lambda: {'omni': [], 'first': [], 'rand': [], 'n': []})
rng = np.random.default_rng(777)
for si, lst in per.items():
    it = data[si]; gt = it['fake_image_box']; c = it['fake_cls']
    lst.sort(key=lambda x: x[0])
    boxes = [b for _, b, _ in lst]
    sc    = np.array([s for _, _, s in lst])
    res[c]['omni'].append(iou(gt, boxes[int(sc.argmax())]))
    res[c]['first'].append(iou(gt, boxes[0]))
    res[c]['rand'].append(iou(gt, boxes[rng.integers(len(boxes))]))
    res[c]['n'].append(len(boxes))

print(f"\n{'类别':<38} {'n':>6} {'脸数':>5} | {'随机':>7} {'第一张':>7} {'★OmniFD':>9}")
print('-'*80)
tot = defaultdict(list)
for c in sorted(res):
    r = res[c]
    for k in ('omni','first','rand'): tot[k] += r[k]
    tot['n'] += r['n']
    print(f"{c:<38} {len(r['omni']):>6} {np.mean(r['n']):>5.2f} | "
          f"{np.mean(r['rand'])*100:>6.2f}% {np.mean(r['first'])*100:>6.2f}% "
          f"{np.mean(r['omni'])*100:>8.2f}%")
print('-'*80)
print(f"{'全体':<38} {len(tot['omni']):>6} {np.mean(tot['n']):>5.2f} | "
      f"{np.mean(tot['rand'])*100:>6.2f}% {np.mean(tot['first'])*100:>6.2f}% "
      f"{np.mean(tot['omni'])*100:>8.2f}%")
print(f"\n参考: ASAP 当前实测 IoU = 74.43%,  上限 = 100%")

np.savez(OUT, scores=scores,
         items=np.array([(i[0], i[1]) for i in items]),
         omni=np.array(tot['omni']), first=np.array(tot['first']))
print(f'\n逐样本数据已保存: {OUT}')
