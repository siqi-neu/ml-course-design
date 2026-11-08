import os, numpy as np, pandas as pd
import torch, torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

BASE      = '/kaggle/input/competitions/2026task1/dataset-for-task2/dataset-for-task2'
TRAIN_DIR = f'{BASE}/train'
TEST_DIR  = f'{BASE}/test'
SUB_CSV   = '/kaggle/input/competitions/2026task1/submission-for-task2.csv'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("device:", device)

IMG_SIZE = 224
BATCH    = 32


train_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(0.2, 0.2, 0.2, 0.05),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])
val_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])

full_ds = datasets.ImageFolder(TRAIN_DIR, transform=train_tf)
print("class_to_idx:", full_ds.class_to_idx)
print("总训练图:", len(full_ds))

targets = [y for _, y in full_ds.samples]
idx_all = np.arange(len(full_ds))
idx_tr, idx_va = train_test_split(idx_all, test_size=0.2,
                                  stratify=targets, random_state=42)

train_ds = Subset(full_ds, idx_tr)
val_base = datasets.ImageFolder(TRAIN_DIR, transform=val_tf)
val_ds   = Subset(val_base, idx_va)

train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True,  num_workers=2)
val_loader   = DataLoader(val_ds,   batch_size=BATCH, shuffle=False, num_workers=2)

class_names = full_ds.classes
NUM_CLASSES = len(class_names)
print("类别:", class_names)


model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, NUM_CLASSES)
model = model.to(device)

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)

def run_epoch(model, loader, train=True):
    model.train() if train else model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    all_pred, all_true = [], []
    torch.set_grad_enabled(train)
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        if train: optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        if train:
            loss.backward(); optimizer.step()
        loss_sum += loss.item() * x.size(0)
        p = out.argmax(1)
        correct += (p == y).sum().item()
        total   += x.size(0)
        all_pred += p.cpu().tolist()
        all_true += y.cpu().tolist()
    return loss_sum/total, correct/total, f1_score(all_true, all_pred, average='macro')

EPOCHS = 20
best_f1, best_state = 0, None
for epoch in range(EPOCHS):
    tl, ta, tf1 = run_epoch(model, train_loader, True)
    vl, va, vf1 = run_epoch(model, val_loader,   False)
    scheduler.step()
    print(f"Epoch {epoch+1:02d} | train loss {tl:.4f} acc {ta:.4f} f1 {tf1:.4f} "
          f"| val loss {vl:.4f} acc {va:.4f} f1 {vf1:.4f}")
    if vf1 > best_f1:
        best_f1 = vf1
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

model.load_state_dict(best_state)
model.to(device)
print("Best val macro-F1:", best_f1)
torch.save(model.state_dict(), '/kaggle/working/effb0_best.pt')


idx_to_class = {v: k for k, v in full_ds.class_to_idx.items()}
test_files = sorted(os.listdir(TEST_DIR))

model.eval()
all_probs = []
with torch.no_grad():
    for fn in test_files:
        img = Image.open(os.path.join(TEST_DIR, fn)).convert('RGB')
        probs = []
        for v in [img,
                  img.transpose(Image.FLIP_LEFT_RIGHT),
                  img.transpose(Image.FLIP_TOP_BOTTOM),
                  img.rotate(180)]:
            x = val_tf(v).unsqueeze(0).to(device)
            probs.append(torch.softmax(model(x), dim=1))
        all_probs.append(torch.stack(probs).mean(0)[0].cpu().numpy())

probs_A = np.stack(all_probs)
np.save('/kaggle/working/test_probs_effb0.npy', probs_A)
print("saved:", probs_A.shape)


print("Best val macro-F1:", best_f1)


import os, numpy as np, pandas as pd
import torch, torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models
from PIL import Image
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score

BASE      = '/kaggle/input/competitions/2026task1/dataset-for-task2/dataset-for-task2'
TRAIN_DIR = f'{BASE}/train'
TEST_DIR  = f'{BASE}/test'
SUB_CSV   = '/kaggle/input/competitions/2026task1/submission-for-task2.csv'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("device:", device)

IMG_SIZE = 224
BATCH    = 32

train_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(0.2, 0.2, 0.2, 0.05),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])
val_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])

full_ds = datasets.ImageFolder(TRAIN_DIR, transform=train_tf)
val_base = datasets.ImageFolder(TRAIN_DIR, transform=val_tf)
print("class_to_idx:", full_ds.class_to_idx)

class_names = full_ds.classes
NUM_CLASSES = len(class_names)
targets = np.array([y for _, y in full_ds.samples])
idx_all = np.arange(len(full_ds))
test_files = sorted(os.listdir(TEST_DIR))
print("类别:", class_names, "| 训练图:", len(full_ds), "| 测试图:", len(test_files))


def train_one_fold(tr_idx, va_idx, fold_seed=0):
    torch.manual_seed(fold_seed)
    np.random.seed(fold_seed)

    train_ds = Subset(full_ds, tr_idx)
    val_ds   = Subset(val_base, va_idx)
    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True,  num_workers=2)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH, shuffle=False, num_workers=2)

    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=15)

    def run_epoch(loader, train=True):
        model.train() if train else model.eval()
        total, correct = 0, 0
        all_pred, all_true = [], []
        torch.set_grad_enabled(train)
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            if train: optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            if train:
                loss.backward(); optimizer.step()
            p = out.argmax(1)
            correct += (p == y).sum().item()
            total   += x.size(0)
            all_pred += p.cpu().tolist()
            all_true += y.cpu().tolist()
        return correct/total, f1_score(all_true, all_pred, average='macro')

    best_f1, best_state = 0, None
    for epoch in range(15):
        ta, tf1 = run_epoch(train_loader, True)
        va, vf1 = run_epoch(val_loader,   False)
        scheduler.step()
        if vf1 > best_f1:
            best_f1 = vf1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    model.to(device)
    return model, best_f1

def predict_proba_tta(model, files, test_dir, tf):
    model.eval()
    all_probs = []
    with torch.no_grad():
        for fn in files:
            img = Image.open(os.path.join(test_dir, fn)).convert('RGB')
            probs = []
            for v in [img,
                      img.transpose(Image.FLIP_LEFT_RIGHT),
                      img.transpose(Image.FLIP_TOP_BOTTOM),
                      img.rotate(180)]:
                x = tf(v).unsqueeze(0).to(device)
                probs.append(torch.softmax(model(x), dim=1))
            all_probs.append(torch.stack(probs).mean(0)[0].cpu().numpy())
    return np.stack(all_probs)


skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
all_test_probs = np.zeros((len(test_files), NUM_CLASSES))
fold_scores = []

for fold, (tr_idx, va_idx) in enumerate(skf.split(idx_all, targets)):
    print(f"\n===== Fold {fold+1}/5 =====")
    model, best_f1 = train_one_fold(tr_idx, va_idx, fold_seed=fold)
    fold_scores.append(best_f1)
    print(f"Fold {fold+1} best val macro-F1: {best_f1:.4f}")

    fold_probs = predict_proba_tta(model, test_files, TEST_DIR, val_tf)
    all_test_probs += fold_probs


    del model
    torch.cuda.empty_cache()

print("\n各折 val F1:", [f"{s:.4f}" for s in fold_scores])
print("平均 val F1:", np.mean(fold_scores))
print("集成后测试集概率 shape:", all_test_probs.shape)
np.save('/kaggle/working/test_probs_5fold.npy', all_test_probs)


idx_to_class = {v: k for k, v in full_ds.class_to_idx.items()}


final_idx = all_test_probs.argmax(1)
preds = [idx_to_class[i] for i in final_idx]

sub = pd.DataFrame({'ID': test_files, 'Category': preds})
sub.to_csv('/kaggle/working/submission_5fold_tta.csv', index=False)


template = pd.read_csv(SUB_CSV)
assert list(sub.columns) == list(template.columns), "列名不匹配"
assert len(sub) == len(template), "行数不匹配"
assert set(sub['ID']) == set(template['ID']), "ID 集合不匹配"

print(sub.head())
print("\n类别分布:")
print(sub['Category'].value_counts())
print("\n✅ submission_5fold_tta.csv 已生成")


probs_A = np.load('/kaggle/working/test_probs_effb0.npy')
probs_B = np.load('/kaggle/working/test_probs_5fold.npy')


w_A, w_B = 0.6, 0.4
final_probs = w_A * probs_A + w_B * probs_B

idx_to_class = {v: k for k, v in full_ds.class_to_idx.items()}
preds = [idx_to_class[i] for i in final_probs.argmax(1)]

sub_final = pd.DataFrame({'ID': test_files, 'Category': preds})
sub_final.to_csv('/kaggle/working/submission.csv', index=False)


template = pd.read_csv(SUB_CSV)
assert list(sub_final.columns) == list(template.columns)
assert len(sub_final) == len(template)
assert set(sub_final['ID']) == set(template['ID'])

print(sub_final.head())
print("\n类别分布:")
print(sub_final['Category'].value_counts())
print("\n✅ submission_fusion.csv 已生成")
