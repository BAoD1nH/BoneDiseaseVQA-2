import torch
from torch.utils.data import DataLoader, random_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from dataset import MedicalVQADataset
import json

with open('label_map_label2idx.json', 'r', encoding='utf-8') as f:
    label2idx = json.load(f)

# load tokenizer & dataset
tokenizer = AutoTokenizer.from_pretrained('vimednli/vihealthbert-w_mlm-ViMedNLI')
full = MedicalVQADataset(
    data_json='dataset/cleaned_output_bonedata.json',
    questions_csv='dataset/question_bonedata.csv',
    image_root='.',
    tokenizer=tokenizer,
    feature_extractor=None,
    label2idx=label2idx
)
train_size = int(0.8 * len(full))
val_size = len(full) - train_size
train_ds, val_ds = random_split(full, [train_size, val_size])

# model
model = AutoModelForSequenceClassification.from_pretrained(
    'vimednli/vihealthbert-w_mlm-ViMedNLI',
    num_labels=len(label2idx)
)

training_args = TrainingArguments(
    output_dir='./bert_checkpoints',
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
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
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

trainer.train()