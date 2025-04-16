import os
import torch
import json
from transformers import AutoTokenizer, AutoFeatureExtractor
from model_arch import BoneDiseaseVQA
from dataset import MedicalVQADataset
from PIL import Image
import csv

# Load label mappings
with open('label_map_label2idx.json', 'r', encoding='utf-8') as f:
    label2idx = json.load(f)

with open('label_map_idx2label.json', 'r', encoding='utf-8') as f:
    idx2label = json.load(f)

# Load the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = BoneDiseaseVQA(
    vision_model_name='microsoft/swinv2-base-patch4-window8-256',
    text_model_name='vimednli/vihealthbert-w_mlm-ViMedNLI',
    answer_classes=list(label2idx.keys()),
).to(device)

model.load_state_dict(torch.load('checkpoints/best_model_9050.pt'))
model.eval()

# Load tokenizer and feature extractor
tokenizer = AutoTokenizer.from_pretrained('vimednli/vihealthbert-w_mlm-ViMedNLI')
feature_extractor = AutoFeatureExtractor.from_pretrained('microsoft/swinv2-base-patch4-window8-256')

def infer(image_path, question):
    # Prepare inputs
    image = Image.open(image_path).convert("RGB")  # Ensure the image is in RGB format
    pixel_values = feature_extractor(images=image, return_tensors="pt").pixel_values.to(device)
    inputs = tokenizer(question, return_tensors="pt", padding=True, truncation=True).to(device)

    with torch.no_grad():
        logits = model(pixel_values, inputs['input_ids'], inputs['attention_mask'])
        # Get the predicted class with sigmoid activation
        preds = torch.argmax(logits, dim=1)
        predicted_label = idx2label[str(preds.item())]

    return predicted_label, preds, logits



# Using multiple question for one image
def infer_multiple_questions(image_path, questions):
    results = {}
    for question in questions:
        predicted_label, _,_ = infer(image_path, question)
        results[question] = {
            'predicted_label': predicted_label,
        }
    return results

# Example usage
if __name__ == "__main__":  
    # Example image path and question
    image_path = 'dataset/img-01001-00002.jpg'  # Replace with your image path
    questions_csv = 'dataset/question_bonedata.csv'  # Path to your questions CSV file
    questions = []
    with open(questions_csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            questions.append(row['\ufeffQuestion'])
    results = infer_multiple_questions(image_path, questions)
    print(results)
    # question = input("Nhập câu hỏi")  # Replace with your question

    # # Perform inference
    # predicted_label,_,_  = infer(image_path, question)
    # print(f"Predicted class: {predicted_label}")    

