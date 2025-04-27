import gradio as gr
import torch
import json
from transformers import AutoTokenizer, AutoFeatureExtractor
from model_arch import BoneDiseaseVQA
from PIL import Image

import logging

# Configure logging
logging.basicConfig(level=logging.INFO, filename="app.log", filemode="w", format="%(asctime)s - %(levelname)s - %(message)s")

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

model.load_state_dict(torch.load('checkpoints/best_model_9050.pt', map_location=device))
model.eval()

# Load tokenizer and feature extractor
tokenizer = AutoTokenizer.from_pretrained('vimednli/vihealthbert-w_mlm-ViMedNLI')
feature_extractor = AutoFeatureExtractor.from_pretrained('microsoft/swinv2-base-patch4-window8-256')

# Inference function
def infer(image, question):
    try:
        # Prepare inputs
        image = image.convert("RGB")
        pixel_values = feature_extractor(images=image, return_tensors="pt").pixel_values.to(device)
        inputs = tokenizer(question, return_tensors="pt", padding=True, truncation=True).to(device)

        with torch.no_grad():
            logits = model(pixel_values, inputs['input_ids'], inputs['attention_mask'])
            preds = torch.argmax(logits, dim=1)
            predicted_label = idx2label[str(preds.item())]

        return predicted_label
    except Exception as e:
        return f"Error during inference: {str(e)}"

# Gradio interface
def gradio_interface(image, question):
    try:
        if image is None:
            logging.warning("No image provided.")
            return "Vui lòng chọn ảnh."
        if question.strip() == "":
            logging.warning("No question provided.")
            return "Vui lòng nhập câu hỏi."
        return infer(image, question)
    except Exception as e:
        logging.error(f"Error in Gradio interface: {str(e)}")
        return f"Error: {str(e)}"
# Create Gradio app
with gr.Blocks() as demo:
    gr.Markdown("## Bone Disease VQA - Visual Question Answering for Medical Images")
    with gr.Row():
        with gr.Column():
            image_input = gr.Image(type="pil", label="Chọn ảnh X-quang")
            question_input = gr.Textbox(label="Nhập câu hỏi", placeholder="Ví dụ: What is the condition of the bone?")
            submit_button = gr.Button("Infer")
        with gr.Column():
            output = gr.Textbox(label="Câu trả lời", interactive=False)

    submit_button.click(gradio_interface, inputs=[image_input, question_input], outputs=output)

# Run the app
if __name__ == "__main__":
    demo.launch()