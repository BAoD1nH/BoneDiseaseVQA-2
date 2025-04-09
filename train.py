import os
import torch
from torch.utils.data import DataLoader, random_split
from transformers import AdamW, get_linear_schedule_with_warmup, AutoTokenizer, AutoFeatureExtractor
from model_arch import BoneDiseaseVQA
from dataset import MedicalVQADataset
import json

# hyperparameters
VISION_MODEL = 'microsoft/swinv2-base-patch4-window8-256'
TEXT_MODEL = 'vimednli/vihealthbert-w_mlm-ViMedNLI'
BATCH_SIZE = 8
EPOCHS = 10
LR = 5e-4

# prepare label mapping
# label2idx = {...}; idx2label = {v:k for k,v in label2idx.items()}

with open('label_map_label2idx.json', 'r', encoding='utf-8') as f:
    label2idx = json.load(f)

with open('label_map_idx2label.json', 'r', encoding='utf-8') as f:
    idx2label = json.load(f)

# dataset and dataloaders
tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL)
feature_extractor = AutoFeatureExtractor.from_pretrained(VISION_MODEL)

full = MedicalVQADataset(
    data_json='data.json',
    questions_csv='questions.csv',
    image_root='.',
    tokenizer=tokenizer,
    feature_extractor=feature_extractor,
    label2idx=label2idx
)

train_size = int(0.8 * len(full))
val_size = len(full) - train_size
train_ds, val_ds = random_split(full, [train_size, val_size])
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

# model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = BoneDiseaseVQA(
    vision_model_name=VISION_MODEL,
    text_model_name=TEXT_MODEL,
    answer_classes=list(label2idx.keys()),
).to(device)

# optimizer & scheduler
optimizer = AdamW(model.parameters(), lr=LR)
total_steps = len(train_loader) * EPOCHS
scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=int(0.1*total_steps), num_training_steps=total_steps
)

# training loop
scaler = torch.amp.GradScaler('cuda') 
best_acc = 0.0
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for batch in train_loader:
        optimizer.zero_grad()
        pixel = batch['pixel_values'].to(device)
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        with torch.amp.autocast('cuda'):
            logits = model(pixel, input_ids, attention_mask)
            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        total_loss += loss.item()
    avg_train_loss = total_loss / len(train_loader)
    # validation\ n    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for batch in val_loader:
            pixel = batch['pixel_values'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            logits = model(pixel, input_ids, attention_mask)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    acc = correct / total
    print(f"Epoch {epoch+1}/{EPOCHS} - train_loss: {avg_train_loss:.4f} - val_acc: {acc:.4f}")
    # save best
    if acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(), os.path.join('checkpoints', 'best_model.pt'))
