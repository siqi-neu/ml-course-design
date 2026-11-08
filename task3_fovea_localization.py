import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "timm"], check=False)

import os, time, math, random, re, contextlib
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, RandomSampler
import torchvision.transforms as T
from sklearn.model_selection import KFold
import timm


IMG_SIZE  = 448
EPOCHS    = 100
BATCH     = 8
LR        = 2e-4
FOLDS     = 5
STEPS_CAP = 40

BACKBONES = ["efficientnet_b0", "resnet18",
             "efficientnet_b0", "resnet18",
             "efficientnet_b0"]


IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
INPUT_ROOT = "/kaggle/input"
WD, PATIENCE, SEED, NUM_WORKERS = 0.05, 30, 42, 0
HM_W  = 0.5
SIGMA = 0.04

OUT_DIR  = "/kaggle/working/fovea"
SUB_PATH = "/kaggle/working/submission.csv"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

random.seed(SEED); np.random.seed(SEED)
torch.manual_seed(SEED)
if DEVICE.type == "cuda":
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = True
os.makedirs(OUT_DIR, exist_ok=True)

BANNER = "=" * 78
print(BANNER)
if DEVICE.type == "cuda":
    print(f"设备 {DEVICE}  ({torch.cuda.get_device_name(0)} x{torch.cuda.device_count()})")
else:
    print(f"设备 {DEVICE}  ← 没 GPU，去右侧 Settings → Accelerator 开一个，否则会跑很久")
print(f"torch {torch.__version__} | timm {timm.__version__}")
print(BANNER)


def random_affine(rng, max_rot=15.0, scale_range=(0.85, 1.15), max_shift=0.06):
    th = np.deg2rad(rng.uniform(-max_rot, max_rot))
    s = rng.uniform(*scale_range)
    t = rng.uniform(-max_shift, max_shift, size=2)
    c, sn = math.cos(th), math.sin(th)
    A = s * np.array([[c, -sn], [sn, c]])
    b = 0.5 - A @ np.array([0.5, 0.5]) + t
    return A, b


def forward_point(p, A, b):
    return A @ np.asarray(p, dtype=np.float64) + b


def pil_affine_data(A, b, S):
    Ai = np.linalg.inv(A)
    bi = -Ai @ b
    return (Ai[0, 0], Ai[0, 1], S * bi[0], Ai[1, 0], Ai[1, 1], S * bi[1])


def hflip_point(p):
    return np.array([1.0 - p[0], p[1]])


AXIS_SUFFIXES = ("_Fovea_X", "_Fovea_Y", "Fovea_X", "Fovea_Y",
                 "_fovea_x", "_fovea_y", "_X", "_Y", "-X", "-Y")


def parse_imageid(s):
    s = str(s).strip()
    for suf in AXIS_SUFFIXES:
        if s.endswith(suf):
            base = s[: -len(suf)].rstrip("_- ")
            if base:
                return base, ("X" if suf.upper().endswith("X") else "Y")
    return s, None


def build_index(paths):
    by_name, by_stem, by_low = {}, {}, {}
    for i, p in enumerate(paths):
        by_name[p.name] = i
        by_stem[p.stem] = i
        by_low[p.name.lower()] = i
        by_low[p.stem.lower()] = i
    return by_name, by_stem, by_low


def resolve_key(base, idxs):
    by_name, by_stem, by_low = idxs
    b = str(base).strip()
    for cand in (b, Path(b).name):
        if cand in by_name:
            return by_name[cand]
        if cand.lower() in by_low:
            return by_low[cand.lower()]
    st = Path(b).stem
    if st in by_stem:
        return by_stem[st]
    if st.lower() in by_low:
        return by_low[st.lower()]
    return None


def digit_map(all_imgs):
    m = {}
    for i, p in enumerate(all_imgs):
        for v in {int(x) for x in re.findall(r"\d+", Path(p).stem)}:
            m.setdefault(v, []).append(i)
    return m


def make_resolver(all_imgs):
    idxs = build_index(all_imgs)
    dm = digit_map(all_imgs)

    def resolve(key):
        i = resolve_key(key, idxs)
        if i is not None:
            return i
        s = str(key).strip()
        if re.fullmatch(r"\d+", s):
            c = dm.get(int(s), [])
            if len(c) == 1:
                return c[0]
        return None

    return resolve


def match_gt_to_images(gt, all_imgs, verbose=True):
    idxs = build_index(all_imgs)
    dm = digit_map(all_imgs)
    keys = [str(k).strip() for k in gt["key"]]
    idx = [resolve_key(k, idxs) for k in keys]
    n_exact = sum(1 for i in idx if i is not None)

    used = {i for i in idx if i is not None}
    n_num = 0
    for k, s in enumerate(keys):
        if idx[k] is not None or not re.fullmatch(r"\d+", s):
            continue
        cand = [i for i in dm.get(int(s), []) if i not in used]
        if len(cand) == 1:
            idx[k] = cand[0]
            used.add(cand[0])
            n_num += 1

    n_miss = sum(1 for i in idx if i is None)
    rep = {"n_total": len(keys), "n_exact": n_exact, "n_numeric": n_num, "n_miss": n_miss}

    if verbose:
        print(f"图片样例   : {[p.name for p in all_imgs[:10]]}")
        print(f"标注样例   : {keys[:10]}")
        shown = 0
        for k, i in enumerate(idx):
            if i is not None and shown < 3:
                print(f"  {keys[k]!r:>14} -> {all_imgs[i].name}")
                shown += 1
        if n_miss:
            miss_keys = [keys[k] for k in range(len(keys)) if idx[k] is None][:6]
            print(f"对不上     : {n_miss} 个，例如 {miss_keys}")

    res = gt.copy()
    res["idx"] = idx
    return res, rep


class CoordSpace:

    def __init__(self, kind, sx=1.0, sy=1.0):
        self.kind, self.sx, self.sy = kind, float(sx), float(sy)

    def to_norm(self, x, y, w, h):
        if self.kind == "norm":
            return x, y
        if self.kind == "pix":
            return x / w, y / h
        return x / self.sx, y / self.sy

    def from_norm(self, xn, yn, w, h):
        if self.kind == "norm":
            return xn, yn
        if self.kind == "pix":
            return xn * w, yn * h
        return xn * self.sx, yn * self.sy

    def describe(self):
        return {"norm": "归一化 [0,1]",
                "pix": "原图像素（按各自宽高归一化）",
                "shared": f"统一缩放空间像素（x/{self.sx:.0f}, y/{self.sy:.0f}）"}[self.kind]


def read_head(p, nrows=5):
    for enc in (None, "latin-1", "utf-8-sig"):
        try:
            return pd.read_csv(p, nrows=nrows) if enc is None else \
                   pd.read_csv(p, nrows=nrows, encoding=enc)
        except Exception:
            continue
    return None


def looks_like_template(df):
    num = df.select_dtypes("number")
    if num.shape[1] == 0:
        return False
    return int(num.nunique(dropna=True).max()) <= 1


def score_as_gt(p, df):
    low = [str(c).lower() for c in df.columns]
    name = p.name.lower()
    s = 0.0
    if "submission" in name or "sample" in name:
        s -= 100
    if "train" in name:
        s += 30
    if any(k in name for k in ("gt", "label", "annot", "ground")):
        s += 20
    if looks_like_template(df):
        s -= 100
    if any("fovea" in c for c in low):
        s += 25
    if len(df.columns) >= 3:
        s += 10
    if len(df) >= 20:
        s += 10
    return s


def find_gt():
    cand_roots = [str(Path(INPUT_ROOT) / "competitions" / "2026task3"), INPUT_ROOT]
    seen, csvs = set(), []
    for r in cand_roots:
        if not Path(r).exists():
            continue
        for p in Path(r).rglob("*.csv"):
            rp = p.resolve()
            if rp in seen:
                continue
            seen.add(rp)
            csvs.append(rp)
    if not csvs:
        raise SystemExit(f"没在 {INPUT_ROOT} 下找到任何 csv，把 input 目录树发我")

    scored = []
    for p in csvs:
        head = read_head(p)
        if head is None or head.empty:
            continue
        scored.append((score_as_gt(p, head), p))
    scored.sort(key=lambda t: -t[0])

    print("\n扫到的 csv：")
    for s, p in scored:
        print(f"  [{s:>5.0f}] {p}")

    if not scored or scored[0][0] < 0:
        raise SystemExit("没认出哪个 csv 是标注表，把上面这串打印发我")
    return scored[0][1]


def key_to_str(s):
    if pd.api.types.is_numeric_dtype(s):
        v = pd.to_numeric(s, errors="coerce")
        if len(v.dropna()) and bool((v.dropna() % 1 == 0).all()):
            return v.astype("Int64").astype(str)
        return v.astype(str)
    return s.astype(str)


def parse_gt(csv_path):
    df = pd.read_csv(csv_path)

    conv = {c: pd.to_numeric(df[c], errors="coerce") for c in df.columns}
    num_cols = [c for c in df.columns if conv[c].notna().mean() >= 0.9]
    str_cols = [c for c in df.columns if c not in num_cols]

    def pick_name_col(pool):
        for key in ("image", "id", "name", "file", "case", "key", "data"):
            for c in pool:
                if key in str(c).lower():
                    return c
        return pool[0] if pool else None

    nc = pick_name_col(str_cols)
    if nc is None:
        nc = pick_name_col(list(df.columns)) or df.columns[0]
        num_cols = [c for c in num_cols if c != nc]


    if len(num_cols) >= 2:
        xc = next((c for c in num_cols if "x" in str(c).lower()), num_cols[0])
        yc = next((c for c in num_cols if "y" in str(c).lower() and c != xc), None)
        if yc is None:
            yc = next((c for c in num_cols if c != xc), None)
        if yc is None:
            raise SystemExit(f"宽表里只认出一个坐标列：{list(df.columns)}")
        out = pd.DataFrame({"key": key_to_str(df[nc]), "x": conv[xc], "y": conv[yc]})
        print(f"标注格式   : 宽表  {csv_path.name}  ({nc} / {xc} / {yc})")
        return out.dropna().reset_index(drop=True)


    vc = num_cols[0] if num_cols else None
    if nc is None or vc is None:
        raise SystemExit(f"标注表解析不了：{list(df.columns)}")
    parsed = [parse_imageid(v) for v in df[nc]]
    if any(ax is None for _, ax in parsed):
        bad = [str(v) for (b, ax), v in zip(parsed, df[nc]) if ax is None][:3]
        raise SystemExit(f"长表里有些 ImageID 认不出是 X 还是 Y，例如 {bad}")
    rec = {}
    for (base, ax), val in zip(parsed, conv[vc]):
        rec.setdefault(base, {})[ax] = val
    rows = [{"key": k, "x": v.get("X"), "y": v.get("Y")} for k, v in rec.items()]
    out = pd.DataFrame(rows).dropna().reset_index(drop=True)
    print(f"标注格式   : 长表  {csv_path.name}  ({nc} / {vc})  "
          f"{len(df)} 行 → {len(out)} 个点")
    return out


GT_CSV = find_gt()
print(f"\n标注文件   : {GT_CSV}")
gt = parse_gt(GT_CSV)
print(f"标注点数   : {len(gt)}")


def list_images():
    probe = GT_CSV.parent
    for _ in range(3):
        imgs, seen = [], set()
        for p in sorted(probe.rglob("*")):
            if p.suffix.lower() in IMG_EXT:
                rp = p.resolve()
                if rp not in seen:
                    seen.add(rp)
                    imgs.append(p)
        if imgs:
            print(f"图片目录   : {probe}  (找到 {len(imgs)} 张)")
            return imgs
        probe = probe.parent
    raise SystemExit(f"在 {GT_CSV.parent} 及上两级都没找到图片，把 input 目录树发我")


ALL_IMGS = list_images()
gt, REP = match_gt_to_images(gt, ALL_IMGS)
print(f"配对结果   : 精确 {REP['n_exact']} / 按数字兜底 {REP['n_numeric']} / "
      f"对不上 {REP['n_miss']}  (共 {REP['n_total']})")
if REP["n_miss"]:
    if REP["n_miss"] > 0.1 * REP["n_total"]:
        raise SystemExit(
            "标注和图片对不上 —— 把上面「图片样例」「标注样例」「对不上」这三行发我，"
            "我按真实命名改匹配规则")
    print("⚠️  少数标注没配上，已跳过")

gt = gt[gt["idx"].notna()].copy()
gt["idx"] = gt["idx"].astype(int)
gt = gt.drop_duplicates("idx").reset_index(drop=True)

TRAIN_PATHS = [ALL_IMGS[i] for i in gt["idx"]]
_train_set = set(gt["idx"].tolist())
TEST_PATHS = [p for i, p in enumerate(ALL_IMGS) if i not in _train_set]

print(f"训练图     : {len(TRAIN_PATHS)}   测试图: {len(TEST_PATHS)}")
if len(TRAIN_PATHS) < 10:
    raise SystemExit("训练图太少了，数据可能没挂上")
if not TEST_PATHS:
    raise SystemExit("没找到测试图，把 input 目录树发我")

def img_size(p):
    with Image.open(p) as im:
        return im.size


_sz = [img_size(p) for p in TRAIN_PATHS]
W_tr = np.array([s[0] for s in _sz], float)
H_tr = np.array([s[1] for s in _sz], float)
gx = gt["x"].to_numpy(float)
gy = gt["y"].to_numpy(float)

print(f"图像尺寸   : W {int(W_tr.min())}~{int(W_tr.max())} (中位 {int(np.median(W_tr))})  "
      f"H {int(H_tr.min())}~{int(H_tr.max())} (中位 {int(np.median(H_tr))})")
print(f"坐标范围   : X {gx.min():.1f}~{gx.max():.1f}   Y {gy.min():.1f}~{gy.max():.1f}")

m = max(gx.max(), gy.max())
if m <= 1.5:
    SPACE = CoordSpace("norm")
elif (gx / W_tr).max() <= 1.02 and (gy / H_tr).max() <= 1.02:
    SPACE = CoordSpace("pix")
else:
    SPACE = CoordSpace("shared", sx=max(gx.max(), 1.0), sy=max(gy.max(), 1.0))
print(f"坐标单位   : {SPACE.describe()}   ← 提交时按这个单位写回去")


if m > 1.5:
    oob = float(np.mean((gx > W_tr * 1.02) | (gy > H_tr * 1.02)))
    if oob > 0.05:
        print(f"⚠️  {oob*100:.0f}% 的标注坐标超出了对应图片的尺寸 —— "
              f"标注和图片可能配错了，先别信这轮结果，把上面几行发我")

_pts = np.array([SPACE.to_norm(x, y, w, h)
                 for x, y, w, h in zip(gx, gy, W_tr, H_tr)], dtype=np.float64)
PTS = np.clip(_pts, 0.0, 1.0)

n_zero = int(((gx == 0) & (gy == 0)).sum())
print(f"归一化坐标 : x {PTS[:,0].min():.3f}~{PTS[:,0].max():.3f}  "
      f"y {PTS[:,1].min():.3f}~{PTS[:,1].max():.3f}")
if n_zero:
    print(f"(0,0) 标注 : {n_zero} 张  ← 「中心凹不可见」，训练时降权处理")


MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
_CACHE = {}


class FoveaDS(Dataset):
    def __init__(self, items, pts, train=False, size=IMG_SIZE, w=None):
        self.items = [str(p) for p in items]
        self.pts = np.asarray(pts, dtype=np.float64)
        self.w = np.ones(len(self.items), np.float32) if w is None \
            else np.asarray(w, np.float32).reshape(-1)
        self.train = train
        self.size = size
        self.cj = T.ColorJitter(0.2, 0.2, 0.1, 0.03) if train else None
        self.to_t = T.ToTensor()
        self.norm = T.Normalize(MEAN, STD)

    def __len__(self):
        return len(self.items)

    def _load(self, p):
        im = _CACHE.get(p)
        if im is None:
            im = Image.open(p).convert("RGB").resize((self.size, self.size), Image.BILINEAR)
            _CACHE[p] = im
        return im

    def __getitem__(self, i):
        p = self.items[i]
        pt = np.array(self.pts[i], dtype=np.float64)
        im = self._load(p)

        if self.train:
            for _ in range(8):
                A, b = random_affine(np.random)
                pt2 = forward_point(pt, A, b)
                if -0.02 <= pt2[0] <= 1.02 and -0.02 <= pt2[1] <= 1.02:
                    im = im.transform((self.size, self.size), Image.AFFINE,
                                      pil_affine_data(A, b, self.size),
                                      resample=Image.BILINEAR, fillcolor=(0, 0, 0))
                    pt = pt2
                    break
            if np.random.rand() < 0.5:
                im = im.transpose(Image.FLIP_LEFT_RIGHT)
                pt = hflip_point(pt)
            im = self.cj(im)

        return (self.norm(self.to_t(im)),
                torch.tensor(pt, dtype=torch.float32),
                torch.tensor(self.w[i], dtype=torch.float32))


class FoveaNet(nn.Module):
    def __init__(self, name, size=IMG_SIZE, drop=0.2):
        super().__init__()
        try:
            self.backbone = timm.create_model(name, pretrained=True, features_only=True)
        except Exception as e:
            print(f"⚠️  {name} 预训练权重加载失败（{type(e).__name__}: {e}）")
            print("    改用随机初始化 —— 分数会明显变差。"
                  "去右侧 Settings 确认 Internet 是 On，然后重跑这一格。")
            self.backbone = timm.create_model(name, pretrained=False, features_only=True)
        self.backbone.eval()
        with torch.no_grad():
            feats = self.backbone(torch.zeros(1, 3, size, size))
        cin = feats[-2 if len(feats) >= 2 else -1].shape[1]
        del feats
        self.head = nn.Sequential(
            nn.Conv2d(cin, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128), nn.SiLU(inplace=True),
            nn.Dropout2d(drop),
            nn.Conv2d(128, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.SiLU(inplace=True),
            nn.Conv2d(64, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.SiLU(inplace=True),
            nn.Conv2d(32, 1, 1),
        )
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)

    def forward(self, x):
        feats = self.backbone(x)
        f = feats[-2] if len(feats) >= 2 else feats[-1]
        f = self.up(f)
        lg = self.head(f)
        B, _, h, w = lg.shape
        p = torch.softmax(lg.reshape(B, -1), 1).reshape(B, 1, h, w)
        xs = (torch.arange(w, device=lg.device, dtype=lg.dtype) + 0.5) / w
        ys = (torch.arange(h, device=lg.device, dtype=lg.dtype) + 0.5) / h
        px = (p.sum(2) * xs).sum(2)
        py = (p.sum(3) * ys).sum(2)
        return torch.cat([px, py], 1), lg


def gauss_target(lg, target):
    h, w = lg.shape[-2], lg.shape[-1]
    dt, dev = lg.dtype, lg.device
    yy = (torch.arange(h, device=dev, dtype=dt) + 0.5).view(1, 1, h, 1)
    xx = (torch.arange(w, device=dev, dtype=dt) + 0.5).view(1, 1, 1, w)
    gy = target[:, 1:2].view(-1, 1, 1, 1) * h
    gx = target[:, 0:1].view(-1, 1, 1, 1) * w
    s = SIGMA * max(h, w)
    return torch.exp(-(((yy - gy) ** 2 + (xx - gx) ** 2) / (2 * s * s)))


def fovea_loss(coords, lg, target, wgt=None):

    coords = coords.float()
    lg = lg.float()
    target = target.float()
    d = F.smooth_l1_loss(coords, target, beta=0.02, reduction="none").mean(1)
    if wgt is not None:
        d = d * wgt.float()
    l1 = d.mean()
    hl = F.binary_cross_entropy_with_logits(lg, gauss_target(lg, target))
    return l1 + HM_W * hl, float(l1), float(hl)


if DEVICE.type == "cuda" and hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
    AUTOCAST = lambda: torch.amp.autocast("cuda")
    try:
        SCALER = torch.amp.GradScaler("cuda")
    except Exception:
        SCALER = torch.cuda.amp.GradScaler()
elif DEVICE.type == "cuda":
    AUTOCAST = lambda: torch.cuda.amp.autocast()
    SCALER = torch.cuda.amp.GradScaler()
else:
    AUTOCAST = contextlib.nullcontext
    SCALER = None


def lr_at(step, total, warm=0.05, mn=0.02):
    if step < total * warm:
        return step / max(1.0, total * warm)
    t = min(1.0, (step - total * warm) / max(1.0, total * (1 - warm)))
    return mn + 0.5 * (1 - mn) * (1 + math.cos(math.pi * t))


@torch.no_grad()
def predict_pts(model, loader):
    model.eval()
    out = []
    for batch in loader:
        x = batch[0].to(DEVICE, non_blocking=True)
        with AUTOCAST():
            c, _ = model(x)
        out.append(c.float().cpu())
    return torch.cat(out).numpy() if out else np.zeros((0, 2), np.float32)


def freeze_bb_bn(model):
    for m in model.backbone.modules():
        if isinstance(m, (nn.BatchNorm2d, nn.SyncBatchNorm)):
            m.eval()


def rmse_pix(pred_n, true_n, W, H):
    pe = np.stack([pred_n[:, 0] * W, pred_n[:, 1] * H], 1)
    te = np.stack([true_n[:, 0] * W, true_n[:, 1] * H], 1)
    return float(np.sqrt(np.mean((pe - te) ** 2)))


W_OF = np.array([s[0] for s in _sz], float)
H_OF = np.array([s[1] for s in _sz], float)


WGT_ALL = np.ones(len(PTS), np.float32)
if n_zero:
    WGT_ALL[np.all(PTS <= 1e-6, axis=1)] = 0.3

splits = list(KFold(n_splits=FOLDS, shuffle=True, random_state=SEED).split(PTS))
fold_rmse, base_rmse, t_start = {}, {}, time.time()

for fold, (tr, va) in enumerate(splits):
    bb = BACKBONES[fold % len(BACKBONES)]
    tr_ds = FoveaDS([TRAIN_PATHS[i] for i in tr], PTS[tr], train=True, w=WGT_ALL[tr])
    va_ds = FoveaDS([TRAIN_PATHS[i] for i in va], PTS[va], train=False)

    dl_kw = dict(num_workers=NUM_WORKERS, pin_memory=(DEVICE.type == "cuda"))
    if len(tr_ds) > STEPS_CAP * BATCH:
        sampler = RandomSampler(tr_ds, replacement=False, num_samples=STEPS_CAP * BATCH)
        tr_dl = DataLoader(tr_ds, batch_size=BATCH, sampler=sampler, **dl_kw)
    else:
        tr_dl = DataLoader(tr_ds, batch_size=BATCH, shuffle=True,
                           drop_last=(len(tr_ds) % BATCH) == 1, **dl_kw)
    va_dl = DataLoader(va_ds, batch_size=BATCH * 2, shuffle=False, **dl_kw)

    print(f"\n{'-'*78}")
    print(f"第 {fold} 折  {bb}  train={len(tr_ds)}  val={len(va_ds)}  "
          f"({len(tr_dl)} steps/epoch)")
    print(f"{'-'*78}")

    model = FoveaNet(bb).to(DEVICE)


    _xb = torch.randn(2, 3, IMG_SIZE, IMG_SIZE, device=DEVICE)
    _yb = torch.tensor([[0.4, 0.5], [0.6, 0.3]], device=DEVICE)
    _wb = torch.ones(2, device=DEVICE)
    model.eval()
    with torch.enable_grad(), AUTOCAST():
        _c, _lg = model(_xb)
        _l, _, _ = fovea_loss(_c, _lg, _yb, _wb)
    assert tuple(_c.shape) == (2, 2), f"输出形状异常 {tuple(_c.shape)}"
    assert float(_c.min()) >= -0.05 and float(_c.max()) <= 1.05, f"输出越界 {_c}"
    _l.backward()
    model.zero_grad(set_to_none=True)
    print(f"自检通过   输出 {tuple(_c.shape)}  热图 {tuple(_lg.shape)}  loss {float(_l):.4f}")

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    total = max(1, len(tr_dl) * EPOCHS)
    step, best, bad = 0, 1e9, 0
    vaW, vaH = W_OF[va], H_OF[va]


    base_p = np.repeat(PTS[tr].mean(0, keepdims=True), len(va), axis=0)
    base_rmse[fold] = rmse_pix(base_p, PTS[va], vaW, vaH)

    for ep in range(EPOCHS):
        model.train()
        freeze_bb_bn(model)
        t0, tot, n_seen = time.time(), 0.0, 0
        for bi, (x, y, w) in enumerate(tr_dl):
            x = x.to(DEVICE, non_blocking=True)
            y = y.to(DEVICE, non_blocking=True)
            w = w.to(DEVICE, non_blocking=True)

            with AUTOCAST():
                coords, lg = model(x)
                loss, l1v, hlv = fovea_loss(coords, lg, y, w)

            opt.zero_grad(set_to_none=True)
            if SCALER is not None:
                SCALER.scale(loss).backward()
                SCALER.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                SCALER.step(opt)
                SCALER.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()

            step += 1
            cur_lr = LR * lr_at(step, total)
            for g in opt.param_groups:
                g["lr"] = cur_lr
            tot += float(loss) * len(x)
            n_seen += len(x)

            if bi % 10 == 0 or bi == len(tr_dl) - 1:
                el = time.time() - t_start
                print(f"  [折{fold}] ep{ep+1:03d}/{EPOCHS} step{bi+1:02d}/{len(tr_dl)} | "
                      f"loss {float(loss):.5f} (L1 {l1v:.4f} HM {hlv:.4f}) | lr {cur_lr:.2e} | "
                      f"已用 {el/60:.0f}min", flush=True)

        va_p = predict_pts(model, va_dl)
        r_now = rmse_pix(va_p, PTS[va], vaW, vaH)
        print(f"[折 {fold}] ep {ep+1:03d}/{EPOCHS} 结束 | train loss {tot/max(1,n_seen):.5f} | "
              f"val RMSE {r_now:7.1f} px | 均值基线 {base_rmse[fold]:7.1f} px | "
              f"{time.time()-t0:.1f}s", flush=True)

        if r_now < best:
            best, bad = r_now, 0
            torch.save(model.state_dict(), os.path.join(OUT_DIR, f"fold{fold}.pt"))
        else:
            bad += 1
            if bad >= PATIENCE:
                print(f"[折 {fold}] 早停 @ ep{ep+1}，best RMSE = {best:.1f} px")
                break

    model.load_state_dict(torch.load(os.path.join(OUT_DIR, f"fold{fold}.pt"), map_location="cpu"))
    model.to(DEVICE)
    fold_rmse[fold] = rmse_pix(predict_pts(model, va_dl), PTS[va], vaW, vaH)
    print(f"[折 {fold}] BEST val RMSE = {fold_rmse[fold]:.1f} px  "
          f"(均值基线 {base_rmse[fold]:.1f} px)")
    del model, opt
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()

print("\n" + BANNER)
for k in sorted(fold_rmse):
    print(f"  折 {k}: 模型 {fold_rmse[k]:7.1f} px   均值基线 {base_rmse[k]:7.1f} px")
mv = float(np.mean(list(fold_rmse.values())))
bv = float(np.mean(list(base_rmse.values())))
print(f"  平均  : 模型 {mv:7.1f} px   均值基线 {bv:7.1f} px")
if mv >= bv:
    print("  ⚠️ 模型没打赢均值基线 —— 多半是坐标单位判错了，把上面的打印发我")
print(BANNER)


class _Flip:

    def __init__(self, base):
        self.base = base

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        x = self.base[i][0]
        return torch.flip(x, dims=[2]), torch.zeros(2), torch.ones(1)


test_ds = FoveaDS(TEST_PATHS, np.zeros((len(TEST_PATHS), 2)), train=False)
dl_test = dict(batch_size=BATCH * 2, shuffle=False, num_workers=NUM_WORKERS,
               pin_memory=(DEVICE.type == "cuda"))
test_dl = DataLoader(test_ds, **dl_test)
flip_dl = DataLoader(_Flip(test_ds), **dl_test)

all_pred = np.zeros((len(TEST_PATHS), 2))
n_used = 0
for fold in range(FOLDS):
    ck = os.path.join(OUT_DIR, f"fold{fold}.pt")
    if not os.path.exists(ck):
        continue
    bb = BACKBONES[fold % len(BACKBONES)]
    m = FoveaNet(bb)
    m.load_state_dict(torch.load(ck, map_location="cpu"))
    m.to(DEVICE).eval()

    p1 = predict_pts(m, test_dl)
    p2 = predict_pts(m, flip_dl)
    p2[:, 0] = 1.0 - p2[:, 0]
    all_pred += (p1 + p2) / 2.0
    n_used += 1
    print(f"  折 {fold} ({bb}) 推理完成")
    del m
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()

if n_used == 0:
    raise SystemExit("没有可用权重，训练阶段没存下任何 fold*.pt")
all_pred /= n_used
print(f"\n集成 {n_used} 折 × 2 视图 TTA")

global_mean = PTS.mean(0)
all_pred = np.nan_to_num(all_pred, nan=float(global_mean[0]))
all_pred = np.clip(all_pred, 0.0, 1.0)


def find_template():
    bases = [GT_CSV.parent] + list(GT_CSV.parent.parents)[:4] + [Path(INPUT_ROOT)]
    for name in ("sample_submission.csv", "submission.csv"):
        for b in bases:
            if (b / name).exists() and (b / name).resolve() != GT_CSV.resolve():
                return b / name
    for b in bases:
        for p in sorted(b.glob("*submission*.csv")):
            if p.resolve() != GT_CSV.resolve():
                return p
    return None


def write_submission(tmpl_path, test_paths, resolve, pred_n, space, w_te, h_te, fill):
    if tmpl_path is None:
        recs = []
        for j, p in enumerate(test_paths):
            x, y = space.from_norm(pred_n[j, 0], pred_n[j, 1], w_te[j], h_te[j])
            recs.append({"ImageID": f"{p.name}_Fovea_X", "value": float(x)})
            recs.append({"ImageID": f"{p.name}_Fovea_Y", "value": float(y)})
        return (pd.DataFrame(recs),
                "没找到提交模板，按 ImageID/value 两行格式输出", 0, [])

    sub = pd.read_csv(tmpl_path)
    print(f"提交模板   : {tmpl_path}   {sub.shape}   列={list(sub.columns)}")
    id_col = sub.columns[0]
    parsed = [parse_imageid(v) for v in sub[id_col]]
    n_miss, miss_samples = 0, []

    if all(ax is not None for _, ax in parsed):

        vcol = next((c for c in sub.columns
                     if str(c).lower() in ("value", "coord", "pred", "coordinate")),
                    sub.columns[-1])
        if vcol == id_col:
            raise SystemExit(f"提交表只有一列，认不出哪列放预测值：{list(sub.columns)}")
        vals = np.array([fill[0] if ax == "X" else fill[1] for _, ax in parsed], float)
        for i, (base, ax) in enumerate(parsed):
            j = resolve(base)
            if j is None:
                n_miss += 1
                if len(miss_samples) < 6:
                    miss_samples.append(str(sub[id_col].iloc[i]))
                continue
            vals[i] = space.from_norm(pred_n[j, 0], pred_n[j, 1],
                                      w_te[j], h_te[j])[0 if ax == "X" else 1]
        sub = sub.copy()
        sub[vcol] = vals
        return sub, f"每坐标一行（{len(sub)} 行，value 列 = {vcol}）", n_miss, miss_samples

    if len(sub.columns) >= 3:

        cols = list(sub.columns)
        xc = next((c for c in cols[1:] if "x" in str(c).lower()), cols[1])
        yc = next((c for c in cols[1:] if "y" in str(c).lower() and c != xc),
                  next(c for c in cols[1:] if c != xc))
        sub = sub.copy()
        for c in (xc, yc):
            sub[c] = pd.to_numeric(sub[c], errors="coerce").astype(float)
        ix, iy = sub.columns.get_loc(xc), sub.columns.get_loc(yc)
        for i in range(len(sub)):
            j = resolve(sub.iloc[i, 0])
            if j is None:
                n_miss += 1
                if len(miss_samples) < 6:
                    miss_samples.append(str(sub.iloc[i, 0]))
                sub.iloc[i, ix], sub.iloc[i, iy] = fill[0], fill[1]
                continue
            x, y = space.from_norm(pred_n[j, 0], pred_n[j, 1], w_te[j], h_te[j])
            sub.iloc[i, ix], sub.iloc[i, iy] = x, y
        return sub, f"一行一图（{len(sub)} 行，X={xc} Y={yc}）", n_miss, miss_samples

    raise SystemExit(f"提交格式没认出来，前 3 行：\n{sub.head(3).to_string()}")


TMPL = find_template()
RESOLVE_T = make_resolver(TEST_PATHS)
W_te = np.array([img_size(p)[0] for p in TEST_PATHS], float)
H_te = np.array([img_size(p)[1] for p in TEST_PATHS], float)
FILL = SPACE.from_norm(global_mean[0], global_mean[1],
                       float(np.median(W_te)), float(np.median(H_te)))

out, fmt_desc, n_miss, miss_samples = write_submission(
    TMPL, TEST_PATHS, RESOLVE_T, all_pred, SPACE, W_te, H_te, FILL)
print(f"提交格式   : {fmt_desc}")
if n_miss:
    print(f"⚠️  {n_miss} 行没匹配上图片，例如 {miss_samples}")
    if n_miss > 0.5 * len(out):
        raise SystemExit("提交模板里的 ID 和测试图对不上 —— 把上面「图片样例」和这几行一起发我")

out.to_csv(SUB_PATH, index=False)
print(f"\n已写出 {SUB_PATH}   shape={out.shape}   空值 {int(out.isna().sum().sum())}")
_num = out.select_dtypes("number")
if _num.shape[1]:
    print(f"数值范围   : {_num.min().min():.2f} ~ {_num.max().max():.2f}")
print(out.head(8).to_string(index=False))
print(f"\n总耗时 {(time.time()-t_start)/60:.1f} 分钟")
print(BANNER)
print("下一步：右上角 Save Version → Save & Run All (Commit)，")
print("        跑完后在 Output / 右侧 Data 面板里取 submission.csv 提交。")
print("        注意 /kaggle/working 不会自动保存，不 Save 会丢。")
