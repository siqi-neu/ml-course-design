import os
import re
import sys
import math
import time
import random
import contextlib
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


INPUT_ROOT = "/kaggle/input"
OUT_DIR    = "/kaggle/working/seg"
SUB_PATH   = "/kaggle/working/submission.csv"

IMG_SIZE    = 512
BATCH       = 4
LR          = 3e-4
WD          = 1e-4
FOLDS       = 5
SEED        = 42


USE_PATCH   = True
PATCH       = 256
PATCH_BATCH = 8
STRIDE      = 128
STEPS_PER_EPOCH = 200
STEPS_PER_FOLD  = 3000


ELASTIC_PROB  = 0.45
ELASTIC_ALPHA = 9.0
ELASTIC_SIGMA = 5.0


CROP_SCALE_LO = 0.70
CROP_SCALE_HI = 1.45


N_SEEDS     = 2
TTA_VIEWS   = 4


TTA_SCALES  = [0.85, 1.0, 1.18]
POST_MIN_PX = 24


CLDICE_W    = 0.5
CLDICE_ITER = 8


MASK_SOFT   = True
EVAL_GT     = "fat"


EPOCHS_MIN   = 40
EPOCHS_MAX   = 240
STEPS_TARGET = 2400
PATIENCE     = 20
TIME_BUDGET  = 30


NUM_WORKERS = 2
BACKBONES   = ("efficientnet_b0", "resnet18")
USE_TTA     = True
USE_GREEN   = True
USE_FOV     = True
LIMIT       = None


RLE_BASE  = 1
RLE_ORDER = "C"
RLE_SIZE  = 512

IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".gif"}

MASK_DIR_HINTS = ("mask", "masks", "label", "labels", "gt", "groundtruth",
                  "ground_truth", "annotation", "annotations", "vessel",
                  "vessels", "seg", "segmentation", "manual", "1st", "2nd")
MASK_STEM_HINTS = ("_mask", "-mask", "_gt", "-gt", "_label", "_labels",
                   "_manual", "_vessel", "_seg", "_segmentation",
                   "mask_", "gt_", "_anno")

BANNER = "=" * 78


def section(t):
    print(f"\n{BANNER}\n{t}\n{BANNER}")


def seed_all(s=SEED):
    random.seed(s)
    np.random.seed(s)
    try:
        import torch
        torch.manual_seed(s)
        torch.cuda.manual_seed_all(s)
    except Exception:
        pass


def _pip(*pkgs):
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", *pkgs],
                       check=False, timeout=900)
    except Exception as e:
        print(f"pip 安装失败：{type(e).__name__}: {e}")


try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
except ImportError:
    _pip("torch")
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader

try:
    import timm
except ImportError:
    _pip("timm")
    try:
        import timm
    except ImportError:
        timm = None

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_amp():
    if DEVICE.type != "cuda":
        return contextlib.nullcontext, None
    try:
        return (lambda: torch.amp.autocast("cuda", enabled=True)), torch.amp.GradScaler("cuda")
    except Exception:
        return (lambda: torch.cuda.amp.autocast()), torch.cuda.amp.GradScaler()


AUTOCAST, SCALER = make_amp()


def img_size(p):
    with Image.open(p) as im:
        return im.size


def is_image(p):
    return Path(p).suffix.lower() in IMG_EXT


def is_mask_path(p):
    p = Path(p)
    dirs = [s.lower() for s in p.parts[-3:-1]]
    if any(any(h in d for h in MASK_DIR_HINTS) for d in dirs):
        return True
    st = p.stem.lower()
    return any(h in st for h in MASK_STEM_HINTS)


def list_images(root, limit_pages=6_000_000):
    out, seen, k = [], set(), 0
    for p in sorted(Path(root).rglob("*")):
        k += 1
        if k > limit_pages:
            print(f"⚠️ 目录太大，扫描被截断在 {limit_pages} 个条目")
            break
        if not p.is_file() or not is_image(p):
            continue
        r = p.resolve()
        if r in seen:
            continue
        seen.add(r)
        out.append(p)
    return out


def build_index(paths):
    paths = [Path(p) for p in paths]
    by_name, by_stem = {}, {}
    for i, p in enumerate(paths):
        for k in {p.name, p.name.lower()}:
            by_name.setdefault(k, []).append(i)
        for k in {p.stem, p.stem.lower()}:
            by_stem.setdefault(k, []).append(i)
    return by_name, by_stem, paths


def _path_affinity(a, b):
    pa, pb = Path(a).parts, Path(b).parts
    n = 0
    for x, y in zip(pa, pb):
        if x != y:
            break
        n += 1
    return n


def _natural_key(name):
    return [int(t) if t.isdigit() else str(t).lower()
            for t in re.split(r"(\d+)", str(name))]


def digit_map(paths):
    m = {}
    for i, p in enumerate(paths):
        for v in {int(x) for x in re.findall(r"\d+", Path(p).stem)}:
            m.setdefault(v, []).append(i)
    return m


def resolve_key(base, idxs, dimap=None, used=None, near=None):
    by_name, by_stem, paths = idxs
    b = str(base).strip()

    def pick(cands):
        c = [i for i in cands if used is None or i not in used]
        if not c:
            return None
        if near is not None and len(c) > 1:
            c.sort(key=lambda i: (-_path_affinity(paths[i], near), i))
        return c[0]

    for key in (b, Path(b).name):
        for k in (key, key.lower()):
            if k in by_name:
                j = pick(by_name[k])
                if j is not None:
                    return j
    for key in (b, Path(b).name):
        st = Path(key).stem
        for k in (st, st.lower()):
            if k in by_stem:
                j = pick(by_stem[k])
                if j is not None:
                    return j
    if dimap is None:
        return None

    def pick_digits(v):
        hit = [i for i in dimap.get(int(v), []) if used is None or i not in used]
        if len(hit) == 1:
            return hit[0]
        if len(hit) > 1 and near is not None:
            return sorted(hit, key=lambda i: (-_path_affinity(paths[i], near), i))[0]
        return None


    if re.fullmatch(r"\d+", b):
        return pick_digits(int(b))

    m = re.match(r"^(\d+)", Path(b).stem)
    if m:
        return pick_digits(int(m.group(1)))
    return None


def key_to_str(series):
    out = []
    for v in series:
        if isinstance(v, float) and float(v).is_integer():
            out.append(str(int(v)))
        elif isinstance(v, (np.floating,)) and float(v).is_integer():
            out.append(str(int(v)))
        else:
            out.append(str(v).strip())
    return out


def strip_mask_suffix(stem):
    s = str(stem).lower()
    for h in sorted(MASK_STEM_HINTS, key=len, reverse=True):
        s = s.replace(h, "")
    return s


def mask_key_candidates(stem):
    s = str(stem)
    raw = [s, strip_mask_suffix(s)]
    raw += re.findall(r"\d+", s)
    raw += re.findall(r"\d+", strip_mask_suffix(s))
    seen, out = set(), []
    for c in raw:
        c = str(c).strip()
        if c and c not in seen:
            seen.add(c)
            out.append(c)
    return out


def score_template(df):
    s = 0.0
    cols = " ".join(str(c).lower() for c in df.columns)
    if len(df.columns) <= 4:
        s += 2.0
    if any(k in cols for k in ("rle", "encoded", "encoding", "pixel",
                               "segmentation", "mask", "pred")):
        s += 3.0
    if any(k in cols for k in ("id", "image", "name", "file")):
        s += 2.0
    if len(df) >= 3:
        s += 0.5
    return s


def find_submission(root, max_csv=2000):
    best, bs, k = None, -1e9, 0
    for p in sorted(Path(root).rglob("*.csv")):
        k += 1
        if k > max_csv:
            break
        try:
            df = pd.read_csv(p, nrows=200)
        except Exception:
            continue
        if df.empty or len(df.columns) < 2:
            continue
        s = score_template(df)
        nl = p.name.lower()
        if any(t in nl for t in ("submission", "sample_sub", "submit")):
            s += 6.0
        if any(t in nl for t in ("train", "gt", "label", "answer", "truth")):
            s -= 4.0
        if s > bs:
            bs, best = s, p
    return best


def count_images(d, cap=400):
    k = 0
    for p in Path(d).rglob("*"):
        if p.is_file() and is_image(p):
            k += 1
            if k >= cap:
                break
    return k


_FCACHE = {}


def cached_submission(d):
    key = str(d)
    if key not in _FCACHE:
        try:
            _FCACHE[key] = find_submission(d)
        except Exception:
            _FCACHE[key] = None
    return _FCACHE[key]


def find_root(root=INPUT_ROOT):
    root = Path(root)
    if not root.exists():
        return root
    cands, seen = [], set()

    def add(c):
        c = Path(c)
        if c.exists() and str(c) not in seen:
            seen.add(str(c))
            cands.append(c)

    for p in root.rglob("*.csv"):
        add(p.parent)
        add(p.parent.parent)
    for d in root.rglob("*"):
        if d.is_dir() and any(h in d.name.lower() for h in ("mask", "label", "gt", "manual")):
            add(d.parent)
            add(d.parent.parent)

    best, bs = root, -1e9
    for c in cands:
        n_img = count_images(c)
        if n_img == 0:
            continue
        s = min(n_img, 200) * 0.01
        if (c / "train").is_dir():
            s += 5
        if any((c / x).is_dir() for x in ("images", "image", "masks", "mask")):
            s += 5
        low = str(c).lower()
        if "task-4" in low or "task4" in low:
            s += 3
        if cached_submission(c) is not None:
            s += 4
        if s > bs:
            bs, best = s, c
    return best


def show_helper_script(root):
    hits = [p for p in Path(root).rglob("*.py")
            if "csv" in p.name.lower() and any(k in p.name.lower()
                                               for k in ("seg", "mask", "rle"))]
    if not hits:
        print("没找到官方 segmentation_to_csv.py —— 按题面例子用 1-based + C 序编码")
        return
    for p in hits[:2]:
        print(f"\n发现官方转换脚本：{p}")
        try:
            txt = p.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            print(f"  读取失败：{e}")
            continue
        lines = txt.splitlines()
        print(f"  （共 {len(lines)} 行，前 60 行如下）")
        for i, l in enumerate(lines[:60], 1):
            print(f"  {i:3d}| {l}")


def clahe_np(gray, clip=2.0, tiles=(8, 8)):
    g = np.ascontiguousarray(gray, dtype=np.uint8)
    H, W = g.shape
    ty = max(1, min(int(tiles[0]), H))
    tx = max(1, min(int(tiles[1]), W))
    ys = np.linspace(0, H, ty + 1).astype(int)
    xs = np.linspace(0, W, tx + 1).astype(int)

    luts = np.zeros((ty, tx, 256), dtype=np.float64)
    for i in range(ty):
        for j in range(tx):
            blk = g[ys[i]:ys[i + 1], xs[j]:xs[j + 1]]
            if blk.size == 0:
                luts[i, j] = np.arange(256, dtype=np.float64)
                continue
            hist = np.bincount(blk.ravel(), minlength=256).astype(np.float64)
            limit = max(1.0, clip * blk.size / 256.0)
            excess = np.maximum(hist - limit, 0.0).sum()
            hist = np.minimum(hist, limit) + excess / 256.0
            cdf = np.cumsum(hist)
            if cdf[-1] <= 0:
                luts[i, j] = np.arange(256, dtype=np.float64)
            else:
                luts[i, j] = cdf / cdf[-1] * 255.0

    cy = (ys[:-1] + ys[1:] - 1) / 2.0
    cx = (xs[:-1] + xs[1:] - 1) / 2.0
    yy = np.arange(H, dtype=np.float64)
    xx = np.arange(W, dtype=np.float64)

    iy = np.clip(np.searchsorted(cy, yy, side="right") - 1, 0, ty - 1)
    ix = np.clip(np.searchsorted(cx, xx, side="right") - 1, 0, tx - 1)
    iy1 = np.clip(iy + 1, 0, ty - 1)
    ix1 = np.clip(ix + 1, 0, tx - 1)

    wy = np.zeros(H, dtype=np.float64)
    dy = cy[iy1] - cy[iy]
    m = dy > 1e-6
    wy[m] = (yy[m] - cy[iy][m]) / dy[m]
    wx = np.zeros(W, dtype=np.float64)
    dx = cx[ix1] - cx[ix]
    m2 = dx > 1e-6
    wx[m2] = (xx[m2] - cx[ix][m2]) / dx[m2]

    wy = np.clip(wy, 0, 1)[:, None]
    wx = np.clip(wx, 0, 1)[None, :]
    pix = g.astype(np.int64)


    v00 = luts[iy[:, None], ix[None, :], pix]
    v01 = luts[iy[:, None], ix1[None, :], pix]
    v10 = luts[iy1[:, None], ix[None, :], pix]
    v11 = luts[iy1[:, None], ix1[None, :], pix]
    out = (v00 * (1 - wy) * (1 - wx) + v01 * (1 - wy) * wx
           + v10 * wy * (1 - wx) + v11 * wy * wx)
    return np.clip(out, 0, 255).astype(np.uint8)


def enhance_rgb(rgb):
    g = np.asarray(rgb, dtype=np.uint8)[:, :, 1]
    if USE_GREEN:
        g = clahe_np(g)
    return np.stack([g, g, g], axis=-1)


def fov_from_rgb(rgb):
    lum = np.asarray(rgb, dtype=np.uint8).max(axis=2)
    m = lum > 12
    if m.mean() < 1e-3:
        return np.ones(lum.shape, np.uint8)
    try:
        from scipy import ndimage
        lab, n = ndimage.label(m)
        if n > 1:

            cnt = np.bincount(lab.ravel(), minlength=n + 1)
            cnt[0] = 0
            m = lab == int(cnt.argmax())
        m = ndimage.binary_fill_holes(m)
    except Exception:
        ys, xs = np.where(m)
        box = np.zeros_like(m)
        if len(ys):
            box[ys.min():ys.max() + 1, xs.min():xs.max() + 1] = True
            m = box
    return m.astype(np.uint8)


def apply_affine(pil, A, t, resample):
    A = np.asarray(A, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64).reshape(2)
    Ai = np.linalg.inv(A)
    ti = -(Ai @ t)
    W, H = pil.size
    a11, a12 = Ai[0, 0], Ai[0, 1]
    a21, a22 = Ai[1, 0], Ai[1, 1]
    A11, A12 = a11, a12 * W / H
    A21, A22 = a21 * H / W, a22
    f0 = W * ti[0] + W / 2.0 - A11 * W / 2.0 - A12 * H / 2.0
    f1 = H * ti[1] + H / 2.0 - A21 * W / 2.0 - A22 * H / 2.0
    return pil.transform((W, H), Image.AFFINE, (A11, A12, f0, A21, A22, f1),
                         resample=resample)


def rand_affine(scale=(0.90, 1.15), deg=12.0, trans=0.05):
    s = random.uniform(*scale)
    th = math.radians(random.uniform(-deg, deg))
    A = s * np.array([[math.cos(th), -math.sin(th)],
                      [math.sin(th), math.cos(th)]])
    t = np.array([random.uniform(-trans, trans), random.uniform(-trans, trans)])
    return A, t


def aug_pair(img, msk, size, train):
    if train:
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            msk = msk.transpose(Image.FLIP_LEFT_RIGHT)
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_TOP_BOTTOM)
            msk = msk.transpose(Image.FLIP_TOP_BOTTOM)
        k = random.randint(0, 3)
        if k:
            img = img.rotate(90 * k, expand=True)
            msk = msk.rotate(90 * k, expand=True)
        A, t = rand_affine()
        img = apply_affine(img, A, t, Image.BILINEAR)
        msk = apply_affine(msk, A, t, Image.NEAREST)
    img = img.resize((size, size), Image.BILINEAR)
    msk = msk.resize((size, size), Image.NEAREST)
    if train:
        b = random.uniform(0.85, 1.15)
        a = random.uniform(-25, 25)
        arr = np.asarray(img, dtype=np.float32) * b + a
        img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    return img, msk


def rle_encode(bin_mask, order=RLE_ORDER, base=RLE_BASE, sep=" "):
    flat = np.asarray(bin_mask).astype(np.uint8).ravel(order=order)
    if flat.size == 0 or int(flat.max()) == 0:
        return ""
    d = np.diff(np.concatenate([[0], flat, [0]]))
    starts = np.flatnonzero(d == 1)
    ends = np.flatnonzero(d == -1)
    lens = ends - starts
    keep = lens > 0
    starts = (starts[keep].astype(np.int64) + base)
    lens = lens[keep].astype(np.int64)
    return sep.join(f"{s} {l}" for s, l in zip(starts.tolist(), lens.tolist()))


def rle_decode(rle, shape, order=RLE_ORDER, base=RLE_BASE):
    n = int(np.prod(shape))
    out = np.zeros(n, dtype=np.uint8)
    s = str(rle).strip()
    if s and s.lower() != "nan":
        v = np.asarray([int(x) for x in s.split() if re.fullmatch(r"-?\d+", x)],
                       dtype=np.int64)
        m = (v.size // 2) * 2
        for a, b in zip(v[0:m:2].tolist(), v[1:m:2].tolist()):
            a = max(0, int(a) - base)
            b = max(0, int(b))
            if b:
                out[a:a + b] = 1
    return out.reshape(shape, order=order)


def detect_fg_value(arr):
    a = np.asarray(arr)
    if a.size == 0:
        return 0, 0.0
    hi = float(a.max())
    if hi <= 1:
        f1 = float((a > 0).mean())
        return (1, f1) if f1 <= 0.5 else (0, 1.0 - f1)
    f_hi = float((a > hi * 0.5).mean())
    return (int(hi), f_hi) if f_hi <= 0.5 else (0, 1.0 - f_hi)


def mask_to_bool(arr, fg_val):
    a = np.asarray(arr, dtype=np.float32)
    if a.size == 0:
        return a
    hi, lo = float(a.max()), float(a.min())
    if hi - lo < 1e-6:
        return np.zeros_like(a, dtype=np.float32)
    mid = 0.5 if hi <= 1.0 else (hi + lo) / 2.0
    if float(fg_val) >= mid:
        return (a >= mid).astype(np.float32)
    return (a < mid).astype(np.float32)


def read_mask_bool(p, fg_val, size=None):
    with Image.open(p) as m:
        if m.mode not in ("L", "I", "F"):
            m = m.convert("L")
        if size is not None:
            m = m.resize(tuple(size), Image.NEAREST)
        arr = np.asarray(m)
    return mask_to_bool(arr, fg_val)


def dice_np(pred_bin, gt_bin, eps=1.0):
    p = np.asarray(pred_bin).astype(np.float32)
    g = np.asarray(gt_bin).astype(np.float32)
    return (2.0 * float((p * g).sum()) + eps) / (float(p.sum()) + float(g.sum()) + eps)


class SegDS(Dataset):

    def __init__(self, items, size=IMG_SIZE, train=False, fg_val=None):
        self.items = list(items)
        self.size = int(size)
        self.train = bool(train)
        self.fg_val = fg_val

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        ip, mp = self.items[i]
        with Image.open(ip) as im:
            rgb = np.asarray(im.convert("RGB"), dtype=np.uint8)
        if USE_FOV:
            fov = fov_from_rgb(rgb)
        else:
            fov = np.ones(rgb.shape[:2], np.uint8)

        img = Image.fromarray(enhance_rgb(rgb))
        if mp is None:
            msk = Image.fromarray(np.zeros((rgb.shape[0], rgb.shape[1]), np.uint8))
        else:
            with Image.open(mp) as m:
                if m.mode not in ("L", "I", "F"):
                    m = m.convert("L")
                arr = np.asarray(m)
            fgv = self.fg_val if self.fg_val is not None else detect_fg_value(arr)[0]
            msk = Image.fromarray((mask_to_bool(arr, fgv) * 255).astype(np.uint8))

        img, msk = aug_pair(img, msk, self.size, self.train)
        fov_img = Image.fromarray((fov * 255).astype(np.uint8)).resize(
            (self.size, self.size), Image.NEAREST)

        x = np.asarray(img, dtype=np.float32) / 255.0
        y = (np.asarray(msk, dtype=np.float32) > 127).astype(np.float32)
        f = (np.asarray(fov_img, dtype=np.float32) > 127).astype(np.float32)
        return (torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1))).float(),
                torch.from_numpy(y[None].copy()).float(),
                torch.from_numpy(f[None].copy()).float())


def slide_positions(L, patch, stride):
    L, patch, stride = int(L), int(patch), int(stride)
    if L <= patch:
        return [0]
    ps = list(range(0, L - patch + 1, max(1, stride)))
    if ps[-1] != L - patch:
        ps.append(L - patch)
    return ps


def taper2d(patch, floor=0.10):
    w = np.hanning(int(patch) + 2)[1:-1].astype(np.float32)
    w = (float(floor) + (1.0 - float(floor)) * w).astype(np.float32)
    return np.outer(w, w)


def load_mask_soft(p, fg_val, size):
    with Image.open(p) as m:
        if m.mode not in ("L", "I", "F"):
            m = m.convert("L")
        arr = np.asarray(m)
    fv = fg_val if fg_val is not None else detect_fg_value(arr)[0]
    b = (mask_to_bool(arr, fv) > 0.5).astype(np.uint8)
    if size is not None:
        im = Image.fromarray((b * 255).astype(np.uint8))
        return np.asarray(im.resize(tuple(size), Image.BILINEAR), dtype=np.uint8)
    return (b * 255).astype(np.uint8)


def soft_to_bin(soft, mode="fat"):
    a = np.asarray(soft)
    return ((a > 0) if str(mode) == "fat" else (a > 127)).astype(np.uint8)


def postprocess_mask(binm, min_px=POST_MIN_PX, fov=None):
    m = np.asarray(binm, dtype=bool)
    if fov is not None:
        m = m & (np.asarray(fov) > 0)
    if int(min_px) <= 0 or not m.any():
        return m.astype(np.uint8)
    try:
        from scipy import ndimage
        lab, k = ndimage.label(m)
        if k > 1:
            cnt = np.bincount(lab.ravel(), minlength=k + 1)
            keep = cnt >= int(min_px)
            keep[0] = False
            m = keep[lab]
    except Exception:
        pass
    return m.astype(np.uint8)


def load_mask_u8(p, fg_val, size):
    with Image.open(p) as m:
        if m.mode not in ("L", "I", "F"):
            m = m.convert("L")
        arr = np.asarray(m)
    fv = fg_val if fg_val is not None else detect_fg_value(arr)[0]
    b = (mask_to_bool(arr, fv) > 0.5).astype(np.uint8)
    if size is not None:
        im = Image.fromarray(b * 255)
        b = (np.asarray(im.resize(tuple(size), Image.NEAREST)) > 127).astype(np.uint8)
    return b


def _gauss_kernel1d(sigma, trunc=3.0):
    r = max(1, int(round(float(trunc) * float(sigma))))
    xx = np.arange(-r, r + 1, dtype=np.float32)
    k = np.exp(-(xx * xx) / (2.0 * float(sigma) ** 2))
    s = float(k.sum())
    return (k / s).astype(np.float32) if s > 0 else np.ones_like(k) / len(k)


def _blur1d(a, k, axis):
    r = len(k) // 2
    pad = [(0, 0)] * a.ndim
    pad[axis] = (r, r)
    ap = np.pad(a, pad, mode="reflect")
    out = np.zeros_like(a)
    n = a.shape[axis]
    for i, kv in enumerate(k):
        sl = [slice(None)] * a.ndim
        sl[axis] = slice(i, i + n)
        out += float(kv) * ap[tuple(sl)]
    return out


def elastic_field(h, w, alpha, sigma):
    h, w = int(h), int(w)
    ds = 4
    hh = max(4, int(math.ceil(h / float(ds))))
    ww = max(4, int(math.ceil(w / float(ds))))
    dy = np.random.uniform(-1.0, 1.0, (hh, ww)).astype(np.float32)
    dx = np.random.uniform(-1.0, 1.0, (hh, ww)).astype(np.float32)
    k = _gauss_kernel1d(max(0.8, float(sigma) / float(ds)))
    dy = _blur1d(_blur1d(dy, k, 0), k, 1)
    dx = _blur1d(_blur1d(dx, k, 0), k, 1)
    if (hh, ww) != (h, w):
        dy = np.asarray(Image.fromarray(dy).resize((w, h), Image.BILINEAR),
                        dtype=np.float32)
        dx = np.asarray(Image.fromarray(dx).resize((w, h), Image.BILINEAR),
                        dtype=np.float32)


    my = float(np.abs(dy).max()) if dy.size else 0.0
    mx = float(np.abs(dx).max()) if dx.size else 0.0
    if my > 1e-8:
        dy = dy / my * float(alpha)
    if mx > 1e-8:
        dx = dx / mx * float(alpha)
    return dy, dx


def _bilinear_index(h, w, dy, dx):
    yy = np.arange(h, dtype=np.float32)[:, None]
    xx = np.arange(w, dtype=np.float32)[None, :]
    sy = np.clip(yy + dy, 0.0, h - 1.0)
    sx = np.clip(xx + dx, 0.0, w - 1.0)
    y0 = np.floor(sy).astype(np.int32)
    x0 = np.floor(sx).astype(np.int32)
    y1 = np.minimum(y0 + 1, h - 1)
    x1 = np.minimum(x0 + 1, w - 1)
    wy = (sy - y0).astype(np.float32)
    wx = (sx - x0).astype(np.float32)
    return y0, x0, y1, x1, wy, wx


def _sample(a, idx):
    y0, x0, y1, x1, wy, wx = idx
    if a.ndim == 3:
        wy = wy[..., None]
        wx = wx[..., None]

    r = (1.0 - wy) * (1.0 - wx) * a[y0, x0]
    r += (1.0 - wy) * wx * a[y0, x1]
    r += wy * (1.0 - wx) * a[y1, x0]
    r += wy * wx * a[y1, x1]
    return r.astype(np.float32, copy=False)


def warp_elastic(img, dy, dx):
    a = np.ascontiguousarray(img, dtype=np.float32)
    h, w = a.shape[:2]
    return _sample(a, _bilinear_index(h, w, dy, dx))


def warp_elastic_pair(x, y, f, dy, dx):
    h, w = x.shape[:2]
    idx = _bilinear_index(h, w, dy, dx)
    xs = np.ascontiguousarray(x, dtype=np.float32)
    same = (xs.ndim == 3 and xs.shape[2] == 3
            and bool((xs[..., 0] == xs[..., 1]).all())
            and bool((xs[..., 1] == xs[..., 2]).all()))
    if same:
        xo = np.repeat(_sample(xs[..., 0], idx)[..., None], 3, axis=2)
    else:
        xo = _sample(xs, idx)
    yo = _sample(np.ascontiguousarray(y, dtype=np.float32), idx)
    fo = _sample(np.ascontiguousarray(f, dtype=np.float32), idx)
    return xo, yo, fo


def crop_patch(x, y, f, patch):
    H, W = x.shape[:2]
    s = int(round(patch * random.uniform(CROP_SCALE_LO, CROP_SCALE_HI)))
    s = int(max(32, min(s, H, W)))
    top = random.randint(0, H - s)
    left = random.randint(0, W - s)
    xc = x[top:top + s, left:left + s]
    yc = y[top:top + s, left:left + s]
    fc = f[top:top + s, left:left + s]
    if s != patch:
        xc = np.asarray(Image.fromarray(xc).resize((patch, patch), Image.BILINEAR))
        yc = np.asarray(Image.fromarray(yc).resize((patch, patch), Image.BILINEAR))
        fc = (np.asarray(Image.fromarray((fc * 255).astype(np.uint8)).resize(
            (patch, patch), Image.NEAREST)) > 127).astype(np.uint8)
    return xc, yc, fc


def aug_patch(x, y, f):
    if random.random() < 0.5:
        x, y, f = x[:, ::-1], y[:, ::-1], f[:, ::-1]
    if random.random() < 0.5:
        x, y, f = x[::-1], y[::-1], f[::-1]
    k = random.randint(0, 3)
    if k:
        x, y, f = np.rot90(x, k), np.rot90(y, k), np.rot90(f, k)


    if ELASTIC_PROB > 0 and random.random() < float(ELASTIC_PROB):
        H, W = x.shape[:2]
        amp = float(ELASTIC_ALPHA) * random.uniform(0.55, 1.0)
        dy, dx = elastic_field(H, W, amp, ELASTIC_SIGMA)
        xw, yw, fw = warp_elastic_pair(x, y, f, dy, dx)
        x = np.clip(xw, 0.0, 255.0).astype(np.uint8)
        y = np.clip(yw, 0.0, 255.0).astype(np.uint8)
        f = (fw > 0.5).astype(np.uint8)
    g = random.uniform(0.70, 1.40)
    b = random.uniform(-0.10, 0.10)
    c = random.uniform(0.85, 1.15)
    v = (np.asarray(x, dtype=np.float32) / 255.0) ** g
    v = (v - 0.5) * c + 0.5 + b
    if random.random() < 0.30:
        v = v + np.random.normal(0.0, random.uniform(0.005, 0.030),
                                 v.shape).astype(np.float32)
    return (np.clip(v, 0.0, 1.0).astype(np.float32),
            np.ascontiguousarray(np.asarray(y, dtype=np.float32) / 255.0),
            np.ascontiguousarray(np.asarray(f, dtype=np.float32)))


class SegCache:

    def __init__(self, items, fg_val, size=IMG_SIZE):
        self.size = int(size)
        self.fg_val = fg_val
        self.x, self.y, self.yb, self.yt, self.f = [], [], [], [], []
        for ip, mp in items:
            with Image.open(ip) as im:
                rgb = np.asarray(im.convert("RGB"), dtype=np.uint8)
            fov = fov_from_rgb(rgb) if USE_FOV else np.ones(rgb.shape[:2], np.uint8)
            img = Image.fromarray(enhance_rgb(rgb)).resize(
                (self.size, self.size), Image.BILINEAR)
            fi = Image.fromarray(
                (np.asarray(fov, dtype=np.float32) * 255).astype(np.uint8))
            fi = fi.resize((self.size, self.size), Image.NEAREST)
            self.x.append(np.asarray(img, dtype=np.uint8))
            self.f.append((np.asarray(fi) > 127).astype(np.uint8))
            if mp is None:
                s = np.zeros((self.size, self.size), np.uint8)
            else:
                s = load_mask_soft(mp, fg_val, (self.size, self.size))
            self.y.append(s)
            self.yb.append(soft_to_bin(s, "fat"))
            self.yt.append(soft_to_bin(s, "thin"))

    def __len__(self):
        return len(self.x)

    def get(self, i):
        return self.x[i], self.y[i], self.f[i]

    def gt(self, i, mode=None):
        return self.yb[i] if str(mode or EVAL_GT) == "fat" else self.yt[i]


class PatchDS(Dataset):

    def __init__(self, cache, idxs, patch=PATCH, n=None):
        self.cache = cache
        self.idxs = list(idxs)
        self.patch = int(patch)
        self.n = int(n) if n else len(self.idxs)

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        j = self.idxs[random.randrange(len(self.idxs))]
        x, y, f = crop_patch(*self.cache.get(j), self.patch)
        x, y, f = aug_patch(x, y, f)
        return (torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1))).float(),
                torch.from_numpy(np.ascontiguousarray(y[None])).float(),
                torch.from_numpy(np.ascontiguousarray(f[None])).float())


def worker_init(wid):
    s = (SEED + 7919 * (int(wid) + 1) + int(time.time() * 1000) % 100000) % (2 ** 31 - 1)
    random.seed(s)
    np.random.seed(s)


@torch.no_grad()
def _predict_one_scale(model, src, size, patch, stride, views):
    tw = taper2d(patch)
    ys = slide_positions(size, patch, stride)
    coords = [(py, px) for py in ys for px in ys]
    canvas = np.zeros((size, size), np.float32)
    wsum = np.zeros((size, size), np.float32)
    for c0 in range(0, len(coords), PATCH_BATCH):
        part = coords[c0:c0 + PATCH_BATCH]
        arr = np.stack([src[py:py + patch, px:px + patch] for py, px in part])

        xb = torch.from_numpy(
            (arr.astype(np.float32) / 255.0).transpose(0, 3, 1, 2)
        ).float().to(DEVICE)
        with AUTOCAST():
            acc = None
            for d in views:
                xi = xb if not d else torch.flip(xb, dims=d)
                pi = torch.sigmoid(model(xi).float())
                if d:
                    pi = torch.flip(pi, dims=d)
                acc = pi if acc is None else acc + pi
            p = acc / float(len(views))
        p = p[:, 0].cpu().numpy()
        for (py, px), pm in zip(part, p):
            canvas[py:py + patch, px:px + patch] += pm * tw
            wsum[py:py + patch, px:px + patch] += tw
    return canvas / np.maximum(wsum, 1e-8)


def predict_slide(model, cache, idxs, patch=PATCH, stride=STRIDE, tta=True,
                  scales=None):
    model.eval()
    size = cache.size
    if scales is None:
        scales = TTA_SCALES if tta else [1.0]
    sc_list = sorted({round(float(s), 4) for s in scales if float(s) > 0}) or [1.0]

    views = [[], [3], [2], [2, 3]][:max(1, min(4, int(TTA_VIEWS)) if tta else 1)]
    out = np.zeros((len(idxs), size, size), np.float32)
    for k, j in enumerate(idxs):
        acc_all = np.zeros((size, size), np.float32)
        for sc in sc_list:
            S = int(max(patch, round(size * sc)))
            if S == size:
                src = cache.x[j]
            else:
                src = np.asarray(
                    Image.fromarray(cache.x[j]).resize((S, S), Image.BILINEAR),
                    dtype=np.uint8)
            p = _predict_one_scale(model, src, S, patch, stride, views)
            if S != size:
                p = np.asarray(
                    Image.fromarray((p * 255).astype(np.uint8)).resize(
                        (size, size), Image.BILINEAR),
                    dtype=np.float32) / 255.0
            acc_all += p
        out[k] = acc_all / float(len(sc_list))
    return out


def save_preview(cache, prob, path, thr=0.5, n=3):
    n = min(int(n), len(cache), len(prob))
    if n <= 0:
        return None
    rows = []
    for k in range(n):
        g = np.asarray(cache.x[k], dtype=np.uint8)
        m = np.asarray(prob[k]) > thr
        ov = g.copy()
        if m.any():
            ov[m] = (0.45 * ov[m].astype(np.float32)
                     + 0.55 * np.array([255, 40, 40], np.float32)).astype(np.uint8)
        rows.append(np.concatenate(
            [g, np.stack([(m * 255).astype(np.uint8)] * 3, -1), ov], 1))
    Image.fromarray(np.concatenate(rows, 0)).save(path)
    return path


class Blk(nn.Module):

    def __init__(self, cin, cout):
        super().__init__()
        g = 8 if cout % 8 == 0 else 1
        self.f = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False),
            nn.GroupNorm(g, cout),
            nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False),
            nn.GroupNorm(g, cout),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.f(x)


class UNet(nn.Module):

    def __init__(self, name="efficientnet_b0", pretrained=True):
        super().__init__()
        if timm is None:
            raise SystemExit("timm 不可用，检查 Kaggle 的 Internet 开关")
        kw = dict(pretrained=pretrained, features_only=True,
                  out_indices=(0, 1, 2, 3, 4))
        ok = True
        try:
            self.enc = timm.create_model(name, **kw)
        except Exception as e:
            print(f"⚠️ {name} 预训练权重加载失败（{type(e).__name__}: {e}）")
            print("   → 改为随机初始化，分数会明显下降；先确认 Internet 已打开")
            kw["pretrained"] = False
            self.enc = timm.create_model(name, **kw)
            ok = False
        self.bn_frozen = ok
        chs = list(self.enc.feature_info.channels())
        self.up = nn.ModuleList()
        cin = chs[-1]
        for c in reversed(chs[:-1]):
            self.up.append(Blk(cin + c, c))
            cin = c
        self.head = nn.Conv2d(cin, 1, 1)


        mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        try:
            import timm.data as _td
            if hasattr(_td, "resolve_model_data_config"):
                cfg = _td.resolve_model_data_config(self.enc)
            else:
                cfg = _td.resolve_data_config(self.enc.pretrained_cfg)
            m = tuple(cfg.get("mean", mean))
            s = tuple(cfg.get("std", std))
            if len(m) == 3 and len(s) == 3 and min(s) > 1e-3:
                mean, std = m, s
        except Exception:
            pass
        self.register_buffer("in_mean", torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer("in_std", torch.tensor(std).view(1, 3, 1, 1))

    def forward(self, x):
        x = (x - self.in_mean) / self.in_std
        feats = self.enc(x)
        d = feats[-1]
        for i, blk in enumerate(self.up):
            skip = feats[-2 - i]
            d = F.interpolate(d, size=skip.shape[-2:], mode="bilinear",
                              align_corners=False)
            d = blk(torch.cat([d, skip], 1))
        d = F.interpolate(d, size=x.shape[-2:], mode="bilinear",
                          align_corners=False)
        return self.head(d)


def set_bn_eval(model):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()


def _soft_erode(img):
    p1 = -F.max_pool2d(-img, (3, 1), (1, 1), (1, 0))
    p2 = -F.max_pool2d(-img, (1, 3), (1, 1), (0, 1))
    return torch.min(p1, p2)


def _soft_dilate(img):
    return F.max_pool2d(img, (3, 3), (1, 1), (1, 1))


def _soft_open(img):
    return _soft_dilate(_soft_erode(img))


def _soft_skel(img, iters=CLDICE_ITER):
    img1 = _soft_open(img)
    skel = F.relu(img - img1)
    for _ in range(int(iters)):
        img = _soft_erode(img)
        img1 = _soft_open(img)
        delta = F.relu(img - img1)
        skel = skel + F.relu(delta - skel * delta)
    return skel


def cldice_loss(pred, tgt, iters=CLDICE_ITER):
    sp = _soft_skel(pred, iters)
    st = _soft_skel(tgt, iters)
    np_ = sp.sum(dim=(1, 2, 3))
    nt_ = st.sum(dim=(1, 2, 3))
    tprec = (sp * tgt).sum(dim=(1, 2, 3)) / (np_ + 1e-4)
    tsens = (st * pred).sum(dim=(1, 2, 3)) / (nt_ + 1e-4)
    cl = 2.0 * tprec * tsens / (tprec + tsens + 1e-6)
    ok = nt_ > 1.0
    if int(ok.sum()) == 0:
        return pred.sum() * 0.0
    return (1.0 - cl[ok]).mean()


def seg_loss(logits, target, fov=None):
    logits = logits.float()
    target = target.float()
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    if fov is not None:
        fov = fov.float()
        bce = (bce * fov).sum() / fov.sum().clamp_min(1.0)
    else:
        bce = bce.mean()
    p = torch.sigmoid(logits)
    pp, tt = (p * fov, target * fov) if fov is not None else (p, target)
    num = 2.0 * (pp * tt).sum(dim=(1, 2, 3)) + 1.0
    den = pp.sum(dim=(1, 2, 3)) + tt.sum(dim=(1, 2, 3)) + 1.0
    dl = 1.0 - (num / den).mean()
    loss = bce + dl
    if CLDICE_W > 0:
        pc, tc = (p * fov, target * fov) if fov is not None else (p, target)
        loss = loss + float(CLDICE_W) * cldice_loss(pc, tc)
    return loss, bce.detach().item(), dl.detach().item()


def lr_at(step, total, warm=0.05, base=1.0):
    if total <= 0:
        return base
    w = max(1, int(total * warm))
    if step <= 0:
        return 0.0
    if step < w:
        return base * step / w
    p = (step - w) / max(1, total - w)
    return base * (0.01 + 0.99 * 0.5 * (1.0 + math.cos(math.pi * min(1.0, p))))


def plan_epochs(n_train, batch=BATCH, target=STEPS_TARGET,
                e_min=EPOCHS_MIN, e_max=EPOCHS_MAX, patience=PATIENCE):
    steps = max(1, math.ceil(int(n_train) / max(1, int(batch))))
    e = int(round(float(target) / steps))
    e = max(int(e_min), min(int(e_max), e))
    return e, max(int(patience), e // 5)


@torch.no_grad()
def predict_prob(model, loader):
    model.eval()
    out = []
    for x, _, _ in loader:
        with AUTOCAST():
            lg = model(x.to(DEVICE, non_blocking=True))
        out.append(torch.sigmoid(lg.float()).cpu())
    if not out:
        return np.zeros((0, IMG_SIZE, IMG_SIZE), np.float32)
    return torch.cat(out).numpy()[:, 0]


def run_epoch(model, loader, opt, epoch, total_ep, log_every=10):
    model.train()
    if getattr(model, "bn_frozen", True):
        set_bn_eval(model)
    tot, bce_s, dl_s, n = 0.0, 0.0, 0.0, 0
    t0 = time.time()
    steps = len(loader)
    g_total = max(1, total_ep * steps)
    for bi, (x, y, f) in enumerate(loader):
        x = x.to(DEVICE, non_blocking=True)
        y = y.to(DEVICE, non_blocking=True)
        f = f.to(DEVICE, non_blocking=True)
        gstep = epoch * steps + bi + 1
        cur_lr = LR * lr_at(gstep, g_total)
        for g in opt.param_groups:
            g["lr"] = cur_lr

        opt.zero_grad(set_to_none=True)
        with AUTOCAST():
            lg = model(x)
            loss, b, d = seg_loss(lg, y, f)

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

        bs = len(x)
        tot += float(loss.detach()) * bs
        bce_s += b * bs
        dl_s += d * bs
        n += bs
        if (bi % log_every == 0) or (bi == steps - 1):
            print(f"  ep{epoch + 1:03d}/{total_ep} step{bi + 1:03d}/{steps} | "
                  f"loss {float(loss.detach()):.4f} (bce {b:.4f} dice {d:.4f}) | "
                  f"lr {cur_lr:.2e} | {(time.time() - t0) / 60:.0f}min", flush=True)
    return tot / max(1, n), bce_s / max(1, n), dl_s / max(1, n)


class _FlipDS(Dataset):

    def __init__(self, base):
        self.base = base

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        x, y, f = self.base[i]
        return torch.flip(x, dims=[2]), y, f


def main():
    section("任务4 · 眼底血管分割")
    print(f"设备 {DEVICE}  |  torch {torch.__version__}  |  "
          f"timm {getattr(timm, '__version__', '未安装')}")
    if DEVICE.type == "cuda":
        print(f"GPU  {torch.cuda.get_device_name(0)}  x{torch.cuda.device_count()}")

    os.makedirs(OUT_DIR, exist_ok=True)
    seed_all()


    root = find_root(INPUT_ROOT)
    print(f"\n数据根目录 : {root}")
    all_imgs = list_images(root)
    print(f"图片总数   : {len(all_imgs)}")
    if not all_imgs:
        raise SystemExit(f"{root} 下一张图都没找到，把 /kaggle/input 目录树发我")

    mask_files = [p for p in all_imgs if is_mask_path(p)]
    img_files = [p for p in all_imgs if not is_mask_path(p)]
    print(f"  疑似掩膜 : {len(mask_files)}   疑似原图 : {len(img_files)}")
    print(f"  原图样例 : {[p.name for p in img_files[:6]]}")
    if mask_files:
        print(f"  掩膜样例 : {[p.name for p in mask_files[:6]]}")
    if not mask_files:
        raise SystemExit(
            "一张掩膜都没认出来。把这两行发我：\n"
            f"  {[str(p) for p in all_imgs[:10]]}")

    show_helper_script(root)


    idxs = build_index(img_files)
    dimap = digit_map(img_files)
    used, pairs, fg_vals, fg_fracs = set(), [], [], []
    for mp in mask_files:
        j = None
        for cand in mask_key_candidates(mp.stem):


            j = resolve_key(cand, idxs, dimap, used, near=mp)
            if j is not None and j not in used:
                break
            j = None
        if j is None:
            continue
        used.add(j)
        pairs.append((img_files[j], mp))
        try:
            with Image.open(mp) as m:
                if m.mode not in ("L", "I", "F"):
                    m = m.convert("L")
                arr = np.asarray(m)
            fg, fr = detect_fg_value(arr)
            fg_vals.append(int(fg))
            fg_fracs.append(float(fr))
        except Exception:
            pass

    if not pairs:
        raise SystemExit(
            "掩膜和原图配不上对。把这两行发我：\n"
            f"  原图样例 {[p.name for p in img_files[:8]]}\n"
            f"  掩膜样例 {[p.name for p in mask_files[:8]]}")

    fg_val = int(np.bincount(np.array(fg_vals, dtype=np.int64)).argmax()) if fg_vals else 0
    print(f"\n配对成功   : {len(pairs)} 对 / {len(mask_files)} 个掩膜")
    for a, b in pairs[:3]:
        print(f"  {a.name}  <-  {b.name}")
    if len(pairs) < 0.8 * len(mask_files):
        print(f"  ⚠️ 有 {len(mask_files) - len(pairs)} 个掩膜没配上，样例：")
        print(f"     {[p.name for p in mask_files[-4:]]}")
    print(f"血管像素值 : {fg_val}   （0 = 白底黑血管；255/1 = 黑底白血管）")
    print(f"血管占比   : {np.mean(fg_fracs) * 100:.2f}%   "
          f"（正常 5%~15%；接近 50% 说明极性判错，把这两行发我）")

    sizes = [img_size(a) for a, _ in pairs]
    Ws = np.array([s[0] for s in sizes])
    Hs = np.array([s[1] for s in sizes])
    print(f"图像尺寸   : W {Ws.min()}~{Ws.max()}  H {Hs.min()}~{Hs.max()}")


    tmpl = find_submission(root)
    sub_df, id_col, val_col = None, None, None
    test_keys, test_paths = [], []
    if tmpl is not None:
        sub_df = pd.read_csv(tmpl)
        id_col = sub_df.columns[0]
        val_col = next((c for c in sub_df.columns[1:]
                        if any(k in str(c).lower() for k in
                               ("rle", "encoded", "encoding", "pixel",
                                "seg", "mask", "pred", "label"))),
                       sub_df.columns[1] if len(sub_df.columns) > 1 else None)
        print(f"\n提交模板   : {tmpl}   {sub_df.shape}   列={list(sub_df.columns)}")
        print("  前 3 行 :")
        for _, r in sub_df.head(3).iterrows():
            print("   ", {k: (str(v)[:38] + ("…" if len(str(v)) > 38 else ""))
                          for k, v in r.items()})

        keys = key_to_str(sub_df[id_col])
        used2 = set(used)
        for k in keys:
            j = resolve_key(k, idxs, dimap, used2)
            if j is not None and j not in used2:
                used2.add(j)
                test_keys.append(k)
                test_paths.append(img_files[j])
            else:
                test_keys.append(k)
                test_paths.append(None)
        n_hit = sum(1 for p in test_paths if p is not None)
        print(f"模板 ID 匹配 : {n_hit}/{len(keys)} 张测试图")
        if n_hit < len(keys):
            miss_i = [i for i, p in enumerate(test_paths) if p is None]
            left = [i for i in range(len(img_files)) if i not in used2]
            print(f"  未匹配 ID  : {[test_keys[i] for i in miss_i][:6]}（共 {len(miss_i)} 个）")
            print(f"  没用上的图 : {[img_files[i].name for i in left[:8]]}（共 {len(left)} 张）")


            if miss_i and len(left) == len(miss_i):
                left.sort(key=lambda i: _natural_key(img_files[i].name))
                for k, i in zip(miss_i, left):
                    test_paths[k] = img_files[i]
                    used2.add(i)
                n_hit = sum(1 for p in test_paths if p is not None)
                print(f"  ⚠️ ID 对不上，按文件名自然顺序把 {len(left)} 张没用上的图"
                      f"依次补给 {len(miss_i)} 个未匹配 ID")
                print(f"     这是顺序假设，不是从文件名推出来的；补完 {n_hit}/{len(keys)}")
                print(f"     补给样例 : {[img_files[i].name for i in left[:6]]}")
        if n_hit < 0.2 * len(keys):
            raise SystemExit(
                "模板里的 ID 和图片几乎全对不上，且数量也对不上，没法兜底。\n"
                f"  模板 ID 样例 {keys[:6]}\n"
                f"  图片名样例 {[p.name for p in img_files[:6]]}\n"
                f"  掩膜样例   {[p.name for p in mask_files[:6]]}\n"
                "把这三行发我。")
    else:
        for i, p in enumerate(img_files):
            if i not in used:
                test_keys.append(p.stem)
                test_paths.append(p)
        print(f"\n没找到提交模板，按「没有掩膜的原图」当测试集：{len(test_paths)} 张")

    n_ok = sum(1 for p in test_paths if p is not None)
    print(f"测试图     : {n_ok} 张可推理 / {len(test_keys)} 行")

    train_items = pairs[:LIMIT] if LIMIT else pairs
    print(f"训练图     : {len(train_items)}")


    n = len(train_items)
    if n < 4:
        raise SystemExit(f"只有 {n} 张训练图，不够训，先把数据确认清楚")
    nf = max(2, min(FOLDS, n // 3))
    nf = min(nf, n)
    print(f"折数       : {nf}  ×  {N_SEEDS} 个种子 = {nf * N_SEEDS} 个模型"
          f"（Dice 是线性指标，多模型概率平均不会更差）")

    order_idx = np.arange(n)
    np.random.RandomState(SEED).shuffle(order_idx)
    folds = np.array_split(order_idx, nf)

    jobs = [(si, fd) for si in range(int(N_SEEDS)) for fd in range(nf)]

    oof_sum = np.zeros((n, IMG_SIZE, IMG_SIZE), np.float32)
    oof_cnt = np.zeros(n, np.float32)
    fold_dice, t_start = {}, time.time()


    cache = None
    if USE_PATCH:
        t_c = time.time()
        cache = SegCache(train_items, fg_val)
        vv_s = float(np.mean([c.mean() / 255.0 for c in cache.y]))
        vv_f = float(np.mean([c.mean() for c in cache.yb]))
        vv_t = float(np.mean([c.mean() for c in cache.yt]))
        print(f"缓存就绪   : {len(cache)} 张 × {IMG_SIZE}²  "
              f"（{time.time() - t_c:.1f}s）")
        print(f"掩膜口径   : 软标签均值 {vv_s * 100:.2f}%  |  "
              f"胖(>0) {vv_f * 100:.2f}%  |  瘦(>127) {vv_t * 100:.2f}%")
        print(f"             榜单口径按胖的算（官方脚本 image[image>0]=1），"
              f"评估/OOF 用 {EVAL_GT}")
        n_ep = max(1, int(math.ceil(STEPS_PER_FOLD / STEPS_PER_EPOCH)))
        pat = max(3, n_ep // 5)
        bs = PATCH_BATCH
    else:
        bs = BATCH
        n_ep, pat = 1, PATIENCE

    for job_i, (si, fold) in enumerate(jobs):
        seed_all(SEED + 1000 * si)
        va = np.sort(folds[fold])
        tr = np.sort(np.concatenate([folds[k] for k in range(nf) if k != fold]))
        bb = BACKBONES[(fold + si) % len(BACKBONES)]
        tag = f"s{si}f{fold}"
        if not USE_PATCH:
            n_ep, pat = plan_epochs(len(tr))
        if USE_PATCH:
            section(f"[{job_i + 1}/{len(jobs)}] 种子 {si} 第 {fold} 折  {bb}  "
                    f"train={len(tr)} val={len(va)}  "
                    f"{n_ep} epoch × {STEPS_PER_EPOCH} 步 = {n_ep * STEPS_PER_EPOCH} 步"
                    f"（每步 {bs} 个 {PATCH}² patch，早停耐心 {pat}）")
        else:
            section(f"[{job_i + 1}/{len(jobs)}] 种子 {si} 第 {fold} 折  {bb}  "
                    f"train={len(tr)} val={len(va)}  "
                    f"{n_ep} epoch × {math.ceil(len(tr) / bs)} 步"
                    f"（早停耐心 {pat}）")

        dl_kw = dict(num_workers=NUM_WORKERS, pin_memory=(DEVICE.type == "cuda"))
        va_dl, va_ds = None, None
        if USE_PATCH:
            tr_ds = PatchDS(cache, tr, PATCH, n=STEPS_PER_EPOCH * bs)
            tr_dl = DataLoader(tr_ds, batch_size=bs, shuffle=True,
                               drop_last=True, worker_init_fn=worker_init,
                               **dl_kw)
        else:
            tr_ds = SegDS([train_items[i] for i in tr], train=True, fg_val=fg_val)
            va_ds = SegDS([train_items[i] for i in va], train=False, fg_val=fg_val)
            tr_dl = DataLoader(tr_ds, batch_size=bs, shuffle=True,
                               drop_last=len(tr) > bs, **dl_kw)
            va_dl = DataLoader(va_ds, batch_size=bs, shuffle=False, **dl_kw)

        model = UNet(bb).to(DEVICE)
        opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)


        model.train()
        if getattr(model, "bn_frozen", True):
            set_bn_eval(model)
        insz = PATCH if USE_PATCH else IMG_SIZE
        _x = torch.rand(2, 3, insz, insz, device=DEVICE)
        _y = (torch.rand(2, 1, insz, insz, device=DEVICE) > 0.85).float()
        _f = torch.ones_like(_y)
        with AUTOCAST():
            _lg = model(_x)
            _l, _b, _d = seg_loss(_lg, _y, _f)
        assert _lg.shape == (2, 1, insz, insz), f"输出形状异常 {tuple(_lg.shape)}"
        assert bool(torch.isfinite(_l)), f"损失非有限值 {_l}"
        assert 0.0 <= _b <= 20.0 and 0.0 <= _d <= 2.0, f"损失异常 bce={_b} dice={_d}"
        _l.backward()
        model.zero_grad(set_to_none=True)
        print(f"自检通过   输出 {tuple(_lg.shape)}  loss {float(_l.detach()):.4f}  "
              f"归一化 mean={[round(float(v), 3) for v in model.in_mean.flatten()]}")

        if USE_PATCH:
            gt_va = np.stack([cache.gt(i, EVAL_GT) for i in va])
        else:
            gt_va = np.stack([read_mask_bool(train_items[i][1], fg_val,
                                            (IMG_SIZE, IMG_SIZE)) for i in va])

        best, best_ep, bad = -1.0, -1, 0
        ck = os.path.join(OUT_DIR, f"{tag}.pt")
        t_fold = time.time()
        for ep in range(n_ep):


            tr_loss, tr_bce, tr_dice = run_epoch(model, tr_dl, opt, ep, n_ep)
            if USE_PATCH:

                prob = predict_slide(model, cache, va, PATCH, STRIDE, tta=False)
            else:
                prob = predict_prob(model, va_dl)
            d = float(np.mean([dice_np(prob[k] > 0.5, gt_va[k])
                               for k in range(len(va))]))
            print(f"[{tag}] ep {ep + 1:03d}/{n_ep}  "
                  f"train loss {tr_loss:.4f} (bce {tr_bce:.4f} dice {tr_dice:.4f})  "
                  f"val Dice {d:.4f}  {(time.time() - t_start) / 60:.0f}min", flush=True)
            if d > best + 1e-5:
                best, best_ep, bad = d, ep, 0
                torch.save(model.state_dict(), ck)
            else:
                bad += 1
                if bad >= pat:
                    print(f"[{tag}] 早停 @ ep{ep + 1}，best Dice = {best:.4f}")
                    break

            if (time.time() - t_fold) / 60 > TIME_BUDGET and ep > 5:
                print(f"[{tag}] 时间预算 {TIME_BUDGET} min 用尽 @ ep{ep + 1}，"
                      f"用当前 best Dice = {best:.4f}")
                break

        model.load_state_dict(torch.load(ck, map_location="cpu"))
        model.to(DEVICE)
        if USE_PATCH:
            oof_sum[va] += predict_slide(model, cache, va, PATCH, STRIDE,
                                         tta=USE_TTA)
        else:
            oof_sum[va] += predict_prob(model, va_dl)
        oof_cnt[va] += 1.0
        fold_dice[tag] = best
        print(f"[{tag}] BEST val Dice = {best:.4f}  (ep {best_ep + 1})")
        del model, opt, tr_dl, tr_ds
        if va_dl is not None:
            del va_dl
        if va_ds is not None:
            del va_ds
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()


    section("阈值搜索（全部折外预测，含后处理）")
    oof_prob = oof_sum / np.maximum(oof_cnt, 1.0)[:, None, None]
    if USE_PATCH:
        gt_fat = np.stack([cache.yb[i] for i in range(n)])
        gt_thin = np.stack([cache.yt[i] for i in range(n)])
        fovs = np.stack([cache.f[i] for i in range(n)])
    else:
        gt_fat = np.stack([read_mask_bool(mp, fg_val, (IMG_SIZE, IMG_SIZE))
                           for _, mp in train_items])
        gt_thin = gt_fat
        fovs = np.ones_like(gt_fat)

    def _score(thr, gtb):
        return float(np.mean([
            dice_np(postprocess_mask(oof_prob[k] > thr, POST_MIN_PX, fovs[k]), gtb[k])
            for k in range(n)]))


    grid = np.round(np.arange(0.10, 0.91, 0.05), 2)
    rows = [(float(t), _score(float(t), gt_fat), _score(float(t), gt_thin))
            for t in grid]
    best_fat = max(rows, key=lambda r: r[1])
    best_thin = max(rows, key=lambda r: r[2])
    i05 = int(np.argmin(np.abs(grid - 0.50)))
    print("  thr     Dice(胖 >0)   Dice(瘦 >127)")
    for t, df, dt in rows:
        mark = ""
        if abs(t - best_fat[0]) < 1e-6:
            mark += "  <- 胖最优"
        if abs(t - best_thin[0]) < 1e-6:
            mark += "  <- 瘦最优"
        print(f"  {t:.2f}    {df:.4f}       {dt:.4f}{mark}")
    print(f"\n胖口径（>0，对齐官方脚本）：阈值 {best_fat[0]:.2f} → OOF Dice {best_fat[1]:.4f}"
          f"（0.5 时 {rows[i05][1]:.4f}，{best_fat[1] - rows[i05][1]:+.4f}）")
    print(f"瘦口径（>127）：阈值 {best_thin[0]:.2f} → OOF Dice {best_thin[1]:.4f}")
    if str(EVAL_GT) == "fat":
        best_thr, best_d, alt_thr = best_fat[0], best_fat[1], best_thin[0]
    else:
        best_thr, best_d, alt_thr = best_thin[0], best_thin[1], best_fat[0]
    print(f"提交用     : 主 {best_thr:.2f}（按 {EVAL_GT} 口径），备用 {alt_thr:.2f}")
    print(f"各模型 best: {[round(v, 4) for v in fold_dice.values()]}")
    print(f"训练耗时 : {(time.time() - t_start) / 60:.0f} 分钟")


    section("测试集推理（多折 + TTA）")
    valid = [(i, test_paths[i]) for i in range(len(test_paths))
             if test_paths[i] is not None]
    if not valid:
        raise SystemExit("没有可推理的测试图")


    cache_te = SegCache([(p, None) for _, p in valid], fg_val)


    _sz = [img_size(p) for _, p in valid]
    _trm = float(np.mean([c.mean() for c in cache.x])) if cache is not None else float("nan")
    _tem = float(np.mean([c.mean() for c in cache_te.x]))
    print(f"测试图尺寸 : W {min(s[0] for s in _sz)}~{max(s[0] for s in _sz)}  "
          f"H {min(s[1] for s in _sz)}~{max(s[1] for s in _sz)}")
    print(f"亮度均值   : 训练 {_trm:.1f}  测试 {_tem:.1f}  "
          f"（差 {abs(_trm - _tem):.1f}；>15 就说明有明显域漂移）")

    n_model = 0
    prob_sum = np.zeros((len(valid), IMG_SIZE, IMG_SIZE), np.float32)
    for si, fold in jobs:
        tag = f"s{si}f{fold}"
        ck = os.path.join(OUT_DIR, f"{tag}.pt")
        if not os.path.exists(ck):
            continue
        bb = BACKBONES[(fold + si) % len(BACKBONES)]
        m = UNet(bb)
        m.load_state_dict(torch.load(ck, map_location="cpu"))
        m.to(DEVICE).eval()
        prob_sum += predict_slide(m, cache_te, list(range(len(valid))),
                                  PATCH, STRIDE, tta=USE_TTA)
        n_model += 1
        print(f"  {tag} ({bb}) 完成", flush=True)
        del m
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()
    prob_mean = prob_sum / max(1, n_model)
    print(f"集成 {n_model} 个模型（{nf} 折 × {N_SEEDS} 种子"
          + (f"，{TTA_VIEWS} 视图 TTA" if USE_TTA else "") + "）")
    pv = save_preview(cache_te, prob_mean, os.path.join(OUT_DIR, "preview_test.png"),
                      thr=best_thr)
    if pv:
        print(f"预览图     : {pv}（左=原图 中=预测 右=叠加；连成一片就是没学到）")


    section("写提交文件")

    def build_rle(thr):
        out, empty, ar = {}, 0, []
        for t, (i, p) in enumerate(valid):
            pm = Image.fromarray((prob_mean[t] * 255).astype(np.uint8)).resize(
                (RLE_SIZE, RLE_SIZE), Image.BILINEAR)
            raw = (np.asarray(pm, dtype=np.float32) / 255.0) > thr
            binm = postprocess_mask(raw, POST_MIN_PX, cache_te.f[t])
            ar.append(float(binm.mean()))
            s = rle_encode(binm)
            if not s:
                empty += 1
            out[test_keys[i]] = s
        return out, empty, ar

    rle_by_key, n_empty, areas = build_rle(best_thr)
    alt_by_key, alt_empty, alt_areas = build_rle(alt_thr)

    probe = next((k for k, v in rle_by_key.items() if v), None)
    if probe is not None:
        back = rle_decode(rle_by_key[probe], (RLE_SIZE, RLE_SIZE))
        n_px = int(back.sum())
        print(f"RLE 自检   : 解回 {RLE_SIZE}×{RLE_SIZE} 得 {n_px} 个血管像素 "
              f"（占 {n_px / (RLE_SIZE * RLE_SIZE) * 100:.2f}%）—— 编码可逆")
        print(f"首行样例   : {rle_by_key[probe][:80]}"
              f"{'…' if len(rle_by_key[probe]) > 80 else ''}")
    print(f"主提交     : 阈值 {best_thr:.2f} → 血管占比均值 {np.mean(areas) * 100:.2f}%  "
          f"空掩膜 {n_empty}/{len(valid)}")
    print(f"备用提交   : 阈值 {alt_thr:.2f} → 血管占比均值 {np.mean(alt_areas) * 100:.2f}%  "
          f"空掩膜 {alt_empty}/{len(valid)}")
    print(f"训练集血管 : 胖(>0) {np.mean([c.mean() for c in cache.yb]) * 100:.2f}%  "
          f"（训练掩膜原始占比 {np.mean(fg_fracs) * 100:.2f}%）")
    if n_empty > 0.3 * len(valid):
        print("  ⚠️ 空掩膜过多 —— 阈值偏高，或掩膜极性判反了")

    def write_sub(rle_map, path, note):
        vals = [rle_map.get(k, "") for k in test_keys]
        if sub_df is not None and val_col is not None:
            out = sub_df.copy()


            out[id_col] = test_keys
            out[val_col] = vals
        else:
            out = pd.DataFrame({"ImageId": test_keys, "EncodedPixels": vals})
        out.to_csv(path, index=False)
        print(f"已写出     : {path}   shape={out.shape}   {note}")
        return out

    out = write_sub(rle_by_key, SUB_PATH, f"阈值 {best_thr:.2f}（{EVAL_GT} 口径）")
    alt_path = str(Path(SUB_PATH).with_name("submission_alt.csv"))
    write_sub(alt_by_key, alt_path, f"阈值 {alt_thr:.2f}（另一口径，备用）")
    print(out.head(5).to_string(index=False, max_colwidth=50))

    section("结果")
    print(f"OOF Dice = {best_d:.4f}   阈值 = {best_thr:.2f}   集成 {n_model} 个模型")
    print(f"两个提交文件：{SUB_PATH}（主） / {alt_path}（备用）")
    print("先交主文件；若 LB 显著低于 OOF（差 0.05 以上），换成备用文件再交一次：")
    print("  两个文件的差别只有阈值，差这么多说明榜单 GT 的胖瘦口径和我推的不一样。")
    print("RLE 已按官方 segmentation_to_csv.py 定死：1-based + C 序 + 512×512。")
    print(BANNER)


if __name__ == "__main__":
    main()
