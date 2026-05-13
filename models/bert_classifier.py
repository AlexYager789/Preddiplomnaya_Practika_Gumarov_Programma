# models/bert_classifier.py
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import sys
import os

sys.path.append(str(Path(__file__).parent.parent))
from config import MODEL_NAME, MAX_LENGTH, MODELS_DIR

class BertClassifier:
    def __init__(self, num_labels=8, model_name=MODEL_NAME):
        self.model_name = model_name
        self.num_labels = num_labels
        self.tokenizer = None
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.label_encoder = None
        print(f"Используется устройство: {self.device}")

    def load_tokenizer(self):
        """Загрузка токенизатора"""
        self.tokenizer = BertTokenizer.from_pretrained(self.model_name)

    def prepare_data(self, df, text_column='lemmatized', label_column='category'):
        """Подготовка данных для обучения"""
        if self.tokenizer is None:
            self.load_tokenizer()

        # Кодируем метки
        self.label_encoder = {label: idx for idx, label in enumerate(df[label_column].unique())}
        joblib.dump(self.label_encoder, MODELS_DIR / 'label_encoder.pkl')
        
        inverse_encoder = {v: k for k, v in self.label_encoder.items()}
        joblib.dump(inverse_encoder, MODELS_DIR / 'inverse_label_encoder.pkl')

        texts = df[text_column].tolist()
        labels = df[label_column].map(self.label_encoder).tolist()

        # Токенизация
        encodings = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=MAX_LENGTH,
            return_tensors='pt'
        )

        return encodings, torch.tensor(labels)

    def build_model(self):
        """Создание модели"""
        self.model = BertForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=self.num_labels,
            problem_type="single_label_classification"
        )
        self.model.to(self.device)

    def train(self, df, epochs=5, batch_size=16, learning_rate=2e-5):
        """Обучение модели"""
        print("Начинается обучение модели...")

        encodings, labels = self.prepare_data(df)
        self.build_model()

        # Разделение на train/test
        train_idx, val_idx = train_test_split(
            range(len(labels)), test_size=0.2, random_state=42, stratify=labels
        )

        train_encodings = {k: v[train_idx] for k, v in encodings.items()}
        val_encodings = {k: v[val_idx] for k, v in encodings.items()}
        train_labels = labels[train_idx]
        val_labels = labels[val_idx]

        optimizer = AdamW(self.model.parameters(), lr=learning_rate)

        self.model.train()
        for epoch in range(epochs):
            print(f"\nЭпоха {epoch+1}/{epochs}")
            total_loss = 0

            # Простой цикл обучения (для небольшого датасета)
            for i in range(0, len(train_labels), batch_size):
                batch_encodings = {k: v[i:i+batch_size].to(self.device) for k, v in train_encodings.items()}
                batch_labels = train_labels[i:i+batch_size].to(self.device)

                optimizer.zero_grad()
                outputs = self.model(**batch_encodings, labels=batch_labels)
                loss = outputs.loss
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / (len(train_labels) / batch_size)
            print(f"Средняя потеря: {avg_loss:.4f}")

        # Сохранение модели
        self.save_model()
        print("Обучение завершено!")

    def save_model(self):
        """Сохранение модели и токенизатора"""
        save_path = MODELS_DIR / "bert_classifier"
        save_path.mkdir(parents=True, exist_ok=True)
        
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        print(f"Модель сохранена в: {save_path}")

    def load_model(self):
        """Загрузка модели"""
        model_path = MODELS_DIR / "bert_classifier"
        if model_path.exists():
            self.tokenizer = BertTokenizer.from_pretrained(model_path)
            self.model = BertForSequenceClassification.from_pretrained(model_path)
            self.model.to(self.device)
            print("Модель успешно загружена")
            return True
        return False

    def predict(self, text: str):
        """Предсказание для одного текста"""
        if self.model is None:
            if not self.load_model():
                return None, 0.0

        cleaned, lemmatized = preprocess_appeal(text)  # Импортируем из utils

        inputs = self.tokenizer(
            lemmatized,
            truncation=True,
            padding=True,
            max_length=MAX_LENGTH,
            return_tensors='pt'
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=1)
            prediction = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][prediction].item()

        # Декодируем метку
        inverse_encoder = joblib.load(MODELS_DIR / 'inverse_label_encoder.pkl')
        predicted_category = inverse_encoder[prediction]

        return predicted_category, confidence


# ====================== Для запуска обучения ======================
if __name__ == "__main__":
    from utils.text_preprocessing import preprocess_appeal
    from data.generate_sample_data import generate_sample_data

    # Генерируем данные, если их нет
    data_path = Path("data/sample_data.xlsx")
    if not data_path.exists():
        print("Генерация синтетического датасета...")
        df = generate_sample_data(400)
    else:
        df = pd.read_excel(data_path)

    classifier = BertClassifier(num_labels=len(df['category'].unique()))
    classifier.train(df, epochs=4, batch_size=12)