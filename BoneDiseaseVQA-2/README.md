# BoneDiseaseVQA-2 Project

## Overview
The BoneDiseaseVQA-2 project is designed for visual question answering (VQA) in the context of bone disease diagnosis. The project utilizes a combination of vision and text models to analyze medical images and answer related questions.

## Project Structure
```
BoneDiseaseVQA-2
├── dataset
│   ├── cleaned_output_bonedata.json  # Cleaned dataset for training
│   ├── question_bonedata.csv          # Questions associated with the dataset
├── checkpoints
│   └── best_model.pt                  # Best-performing model after training
├── src
│   ├── train.py                        # Training loop for the model
│   ├── infer.py                        # Inference script for predictions
│   ├── model_arch.py                   # Model architecture definition
│   └── dataset.py                      # Dataset handling and preprocessing
├── label_map_label2idx.json           # Mapping of labels to indices
├── label_map_idx2label.json           # Mapping of indices back to labels
└── README.md                           # Project documentation
```

## Files Description
- **dataset/cleaned_output_bonedata.json**: Contains the cleaned dataset used for training the model.
- **dataset/question_bonedata.csv**: Contains the questions associated with the bone disease data.
- **checkpoints/best_model.pt**: The saved state of the best-performing model after training.
- **src/train.py**: Contains the training loop for the model, including data loading, model training, validation, and early stopping logic.
- **src/infer.py**: Script to perform inference using the `best_model.pt`. It loads the model, prepares input data, and outputs predictions.
- **src/model_arch.py**: Defines the architecture of the `BoneDiseaseVQA` model, including the vision and text components.
- **src/dataset.py**: Contains the `MedicalVQADataset` class, which handles loading and preprocessing the dataset for training and inference.
- **label_map_label2idx.json**: Maps labels to their corresponding indices.
- **label_map_idx2label.json**: Maps indices back to their corresponding labels.

## Usage
1. **Training the Model**: Run `src/train.py` to train the model using the provided dataset.
2. **Performing Inference**: Use `src/infer.py` to load the trained model and make predictions on new data.

## Requirements
- Python 3.x
- PyTorch
- Transformers
- Other dependencies as specified in the project files.

## License
This project is licensed under the MIT License.