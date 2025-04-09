import os
import json
import random
from PIL import Image
import torch
from torch.utils.data import Dataset
import csv

class MedicalVQADataset(Dataset):
    """
    Dataset for Medical VQA: returns (pixel_values, input_ids, attention_mask, label)
    """
    def __init__(
        self,
        data_json: str,
        questions_csv: str,
        image_root: str,
        tokenizer,
        feature_extractor,
        label2idx: dict,
        transforms=None,
    ):
        super().__init__()
        with open(data_json, 'r', encoding='utf-8') as f:
            self.data = json.load(f)

        # load questions
        self.questions = []
        with open(questions_csv, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.questions.append(row['\ufeffQuestion'])

        # Print all questions
        # print("Questions:")
        # for q in self.questions:
        #     print(q)
    
        self.image_root = image_root
        self.tokenizer = tokenizer
        self.feature_extractor = feature_extractor
        self.label2idx = label2idx
        self.transforms = transforms

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        # load image
        img_path = os.path.join(self.image_root, sample['image_url'])
        image = Image.open(img_path).convert('RGB')

        if self.transforms:
            pixel_values = self.transforms(image)
        else:
            if self.feature_extractor is None:
                pixel_values = None
            else:
                px = self.feature_extractor(images=image, return_tensors='pt')
                pixel_values = px.pixel_values.squeeze(0)

        # random question
        question = random.choice(self.questions)

        
        if self.tokenizer is None:
            text = None
            input_ids = None
            attention_mask = None
        else:
            text = self.tokenizer(
                question,
                padding='max_length',
                truncation=True,
                max_length=512,
                return_tensors='pt'
            )

            input_ids = text.input_ids.squeeze(0)
            attention_mask = text.attention_mask.squeeze(0)

        answer_text = f"{sample['diagnose']}, tình trạng {sample['condition']}"
        print(f"Answer text: {answer_text}")

        # if answer_text not in self.label2idx:
        #     print(f"[UNKNOWN LABEL] '{answer_text}' at index {idx}")
        #     raise KeyError(f"Label '{answer_text}' not found in label2idx.")
        # else:
        label = self.label2idx[answer_text]

        return {
            'pixel_values': pixel_values,
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'label': torch.tensor(label, dtype=torch.long)
        }

if __name__ == "__main__":
    # Print some samples from the dataset
    with open('label_map_label2idx.json', 'r', encoding='utf-8') as f:
        label2idx = json.load(f)

    dataset = MedicalVQADataset(
        data_json='dataset/cleaned_output_bonedata.json',
        questions_csv='dataset/question_bonedata.csv',
        image_root='/kaggle/input/bonevqa/DemoBoneData',
        tokenizer=None,
        feature_extractor=None,
        label2idx=label2idx
    )
    for i in range(5):
        sample = dataset[i]
        print(f"Sample {i}:")
        print(f"Image path: {sample['pixel_values']}")
        print(f"Input IDs: {sample['input_ids']}")
        print(f"Attention Mask: {sample['attention_mask']}")
        print(f"Label: {sample['label']}")
        print()