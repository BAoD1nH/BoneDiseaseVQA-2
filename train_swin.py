import torch
from torch.utils.data import DataLoader, random_split
from transformers import AutoFeatureExtractor, SwinForImageClassification, TrainingArguments, Trainer
from dataset import MedicalVQADataset
import json
# load label2idx from dataset
# e.g., label2idx = {'answer1':0, 'answer2':1, ...}

feature_extractor = AutoFeatureExtractor.from_pretrained('microsoft/swinv2-base-patch4-window8-256')


with open('label_map_label2idx.json', 'r', encoding='utf-8') as f:
    label2idx = json.load(f)

with open('label_map_idx2label.json', 'r', encoding='utf-8') as f:
    idx2label = json.load(f)

# simple image-only dataset wrapper
class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, base_dataset):
        self.base = base_dataset
    def __len__(self): return len(self.base)
    def __getitem__(self, idx):
        d = self.base[idx]
        return {'pixel_values': d['pixel_values'], 'labels': d['label']}

# prepare dataset
full = MedicalVQADataset(
    data_json='dataset/cleaned_output_bonedata.json',
    questions_csv='dataset/question_bonedata.csv',
    image_root='/kaggle/input/boneVQA/',
    tokenizer=None,
    feature_extractor=feature_extractor,
    label2idx=label2idx
)
train_size = int(0.8 * len(full))
val_size = len(full) - train_size

train_ds, val_ds = random_split(full, [train_size, val_size])
train_ds = ImageDataset(train_ds)
val_ds = ImageDataset(val_ds)

model = SwinForImageClassification.from_pretrained(
    'microsoft/swinv2-base-patch4-window8-256',
    num_labels=len(label2idx)
)

training_args = TrainingArguments(
    output_dir='./swin_checkpoints',
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=5,
    evaluation_strategy='epoch',
    save_strategy='epoch',
    logging_strategy='steps',
    logging_steps=50,
    fp16=True,
)

def compute_metrics(p):
    preds = p.predictions.argmax(-1)
    labels = p.label_ids
    acc = (preds == labels).mean()
    return {'accuracy': acc}

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    tokenizer=feature_extractor,
    compute_metrics=compute_metrics
)

trainer.train()