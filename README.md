# Transformer-Based NLP Classification System

This project demonstrates a production-style machine learning workflow for training and serving transformer-based NLP models using the Hugging Face ecosystem and PyTorch.

The project implements an end-to-end pipeline including dataset processing, tokenization, transformer fine-tuning, and real-time inference using an API.

This workflow reflects common machine learning engineering practices used in modern AI platforms and cloud ML environments.

---

## Key Objectives

- Implement transformer-based NLP modeling using Hugging Face Transformers  
- Demonstrate fine-tuning of pretrained language models for text classification  
- Build a reproducible training pipeline using PyTorch and Hugging Face Trainer  
- Deploy a trained model using an API for real-time inference  
- Illustrate ML engineering workflows including dataset preprocessing and model serving  

---

## Technologies Used

### Machine Learning
- PyTorch  
- Hugging Face Transformers  
- Hugging Face Datasets  

### Modeling Techniques
- Transfer Learning  
- Transformer Fine-Tuning  
- WordPiece Tokenization  
- Sequence Classification  

### ML Engineering Tools
- FastAPI for model serving  
- Python for data preprocessing and training pipelines  

---

## Model Architecture

The model used in this project is a pretrained **BERT (Bidirectional Encoder Representations from Transformers)** architecture.

Model used:

bert-base-uncased

BERT uses a transformer encoder architecture with self-attention mechanisms to generate contextual representations of text.

The pretrained model is fine-tuned for binary sentiment classification.

---

## Dataset

Dataset used:

IMDB Movie Reviews Dataset

The dataset contains labeled movie reviews used for sentiment classification.

Data pipeline steps:

1. Load dataset using Hugging Face Datasets
2. Tokenize text using AutoTokenizer
3. Apply padding and truncation
4. Split training and evaluation data

---

## Training Pipeline

The training pipeline performs the following steps:

- Load dataset using Hugging Face Datasets  
- Tokenize text using AutoTokenizer  
- Load pretrained transformer model using AutoModelForSequenceClassification  
- Configure training parameters with TrainingArguments  
- Train model using the Hugging Face Trainer API  

Training parameters used:

- Learning rate: 2e-5  
- Batch size: 8  
- Epochs: 1  
- Evaluation strategy: epoch  

---

## Model Inference API

A lightweight inference API is implemented using **FastAPI**.

The API loads a transformer pipeline and exposes a prediction endpoint.

Endpoint:

GET /predict

Example request:

/predict?text=This movie was amazing

Example response:

{
 "prediction": [
   {
     "label": "POSITIVE",
     "score": 0.98
   }
 ]
}

This demonstrates how transformer models can be deployed as **real-time inference services**.

---

## Repository Structure

transformer-text-classification

train_model.py → transformer model training pipeline  
inference_api.py → FastAPI inference service  
requirements.txt → project dependencies  
README.md → project documentation  

---

## Installation

Install dependencies:

pip install -r requirements.txt

---

## Train the Model

Run the training pipeline:

python train_model.py

---

## Start the Inference API

Start the API server:

uvicorn inference_api:app --reload

Then open:

http://127.0.0.1:8000/predict?text=This movie was fantastic

---

## Machine Learning Concepts Demonstrated

- Transformer-based NLP modeling  
- Contextual embeddings using pretrained models  
- Transfer learning for text classification  
- End-to-end ML pipeline development  
- API-based model serving  

---

## Future Improvements

Potential extensions include:

- Parameter-efficient fine-tuning using PEFT / LoRA  
- Distributed training using Hugging Face Accelerate  
- Containerization using Docker  
- Cloud training using AWS SageMaker or GCP Vertex AI  
- Model monitoring and evaluation pipelines  

---

## Author

Varsha C  
Machine Learning Engineer
