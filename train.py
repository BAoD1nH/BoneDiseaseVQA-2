import os
import torch
from torch.utils.data import DataLoader, random_split
from transformers import AdamW, get_linear_schedule_with_warmup, AutoTokenizer, AutoFeatureExtractor
from model_arch import BoneDiseaseVQA
from dataset import MedicalVQADataset
from tqdm import tqdm
import json
from huggingface_hub import hf_hub_download

# hyperparameters
VISION_MODEL = 'microsoft/swinv2-base-patch4-window8-256'
TEXT_MODEL = 'vimednli/vihealthbert-w_mlm-ViMedNLI'

BATCH_SIZE = 6
EPOCHS = 10
LR = 1e-5
PATIENCE = 3  # early stopping patience

# prepare label mapping
with open('label_map_label2idx.json', 'r', encoding='utf-8') as f:
    label2idx = json.load(f)

with open('label_map_idx2label.json', 'r', encoding='utf-8') as f:
    idx2label = json.load(f)

# dataset and dataloaders
tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL)
feature_extractor = AutoFeatureExtractor.from_pretrained(VISION_MODEL)

full = MedicalVQADataset(
    data_json='dataset/cleaned_output_bonedata.json',
    questions_csv='dataset/question_bonedata.csv',
    image_root='/kaggle/input/bonevqa/DemoBoneData',
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

# # Locally Load weight from best model if exists
# if os.path.exists('checkpoints/best_model.pt'):
#     model.load_state_dict(torch.load('checkpoints/best_model.pt'))
#     print("✅ Model weights loaded from best model.")

# Huggingface load weight from best model
model_path = hf_hub_download(repo_id="Vantuk/BoneDiseaseVQA", filename="best_model_8840.pt")
model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))


# freeze/unfreeze utilities
def freeze_encoder(model):
    for param in model.vision.parameters():
        param.requires_grad = False
    for param in model.text.parameters():
        param.requires_grad = False

def unfreeze_encoder(model):
    for param in model.vision.parameters():
        param.requires_grad = True
    for param in model.text.parameters():
        param.requires_grad = True

# optimizer & scheduler
optimizer = AdamW(model.parameters(), lr=LR)
total_steps = len(train_loader) * EPOCHS
scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps
)

# training loop
scaler = torch.cuda.amp.GradScaler()
best_acc = 0.0
epochs_no_improve = 0
os.makedirs('checkpoints', exist_ok=True)

for epoch in range(EPOCHS):
    # freeze for first 3 epochs
    # if epoch == 0:
    #     freeze_encoder(model)
    #     print("🔒 Encoders frozen.")
    # elif epoch == 3:
    #     unfreeze_encoder(model)
    #     print("🔓 Encoders unfrozen.")

    model.train()
    total_loss = 0
    train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]")
    for batch in train_bar:
        optimizer.zero_grad()
        pixel = batch['pixel_values'].to(device)
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        with torch.cuda.amp.autocast():
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
        train_bar.set_postfix(loss=loss.item())

    avg_train_loss = total_loss / len(train_loader)

    # validation
    model.eval()
    correct, total = 0, 0
    val_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Val]")
    with torch.no_grad():
        for batch in val_bar:
            pixel = batch['pixel_values'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            logits = model(pixel, input_ids, attention_mask)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    acc = correct / total
    print(f"✅ Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f} | Val Acc = {acc:.4f}")

    # Save best model
    if acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(), os.path.join('checkpoints', 'best_model.pt'))
        print("📌 Best model saved.")
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1
        print(f"⚠️ No improvement. ({epochs_no_improve}/{PATIENCE})")

    # Early stopping
    if epochs_no_improve >= PATIENCE:
        print(f"⏹️ Early stopping triggered at epoch {epoch+1}.")
        break
