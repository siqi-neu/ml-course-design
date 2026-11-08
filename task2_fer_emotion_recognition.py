import subprocess
import sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "transformers"], check=False)

import os, numpy as np, pandas as pd
import torch
from PIL import Image
from transformers import ViTImageProcessor, ViTForImageClassification
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("device:", device)

TEST_DIR = '/kaggle/input/competitions/2026task2/fer_data/fer_data/test'
SUB_CSV = '/kaggle/input/competitions/2026task2/submission.csv'
test_files = sorted(os.listdir(TEST_DIR))


print("加载模型 abhilash88/face-emotion-detection ...")
processor = ViTImageProcessor.from_pretrained('abhilash88/face-emotion-detection')
model = ViTForImageClassification.from_pretrained('abhilash88/face-emotion-detection').to(device)
model.eval()


all_probs = []
with torch.no_grad():
    for fn in tqdm(test_files, desc="推理中"):
        img = Image.open(os.path.join(TEST_DIR, fn)).convert('RGB')
        inputs = processor(images=img, return_tensors="pt").to(device)


        logits = model(**inputs).logits
        probs_1 = torch.softmax(logits, dim=1)


        img_flip = img.transpose(Image.FLIP_LEFT_RIGHT)
        inputs_flip = processor(images=img_flip, return_tensors="pt").to(device)
        logits_flip = model(**inputs_flip).logits
        probs_2 = torch.softmax(logits_flip, dim=1)


        prob = ((probs_1 + probs_2) / 2.0).cpu().numpy()[0]
        all_probs.append(prob)

all_probs = np.array(all_probs)


target_probs = all_probs[:, [0, 2, 3, 4, 5, 6]]


pred_task_ids = target_probs.argmax(1).tolist()


sub = pd.DataFrame({'ID': test_files, 'Emotion': pred_task_ids})
sub.to_csv('/kaggle/working/submission.csv', index=False)


template = pd.read_csv(SUB_CSV)
assert list(sub.columns) == list(template.columns), "列名不匹配"
assert len(sub) == len(template), "行数不匹配"
assert set(sub['ID']) == set(template['ID']), "ID不匹配"
assert sub['Emotion'].isin([0,1,2,3,4,5]).all(), "标签越界"

print("\n✅ 生成 submission_abhilash.csv 完毕")
print(sub.head())
print("\n预测类别分布:")
print(sub['Emotion'].value_counts().sort_index())
