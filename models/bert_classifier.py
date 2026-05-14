# models/bert_classifier.py
import torch
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
import pandas as pd
import joblib
from pathlib import Path
import sys
import time

sys.path.append(str(Path(__file__).parent.parent))

from config import MODEL_NAME, MAX_LENGTH, MODELS_DIR
from utils.text_preprocessing import preprocess_appeal


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
        if self.tokenizer is None:
            self.tokenizer = BertTokenizer.from_pretrained(self.model_name)

    def build_model(self):
        self.model = BertForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=self.num_labels
        )
        self.model.to(self.device)

    def train(self, df, epochs=3, batch_size=12):
        """Обучение модели"""
        print("Начинается обучение...")

        self.load_tokenizer()
        self.build_model()

        # Подготовка данных
        texts = df['lemmatized'].tolist()
        labels = df['category'].unique()
        self.label_encoder = {label: idx for idx, label in enumerate(labels)}
        inverse_encoder = {v: k for k, v in self.label_encoder.items()}

        joblib.dump(self.label_encoder, MODELS_DIR / 'label_encoder.pkl')
        joblib.dump(inverse_encoder, MODELS_DIR / 'inverse_label_encoder.pkl')

        label_ids = df['category'].map(self.label_encoder).values

        optimizer = AdamW(self.model.parameters(), lr=2e-5)

        self.model.train()
        for epoch in range(epochs):
            print(f"Эпоха {epoch+1}/{epochs}")
            total_loss = 0
            start_time = time.time()

            for i in range(0, len(df), batch_size):
                batch_texts = texts[i:i+batch_size]
                batch_labels = torch.tensor(label_ids[i:i+batch_size]).to(self.device)

                encodings = self.tokenizer(
                    batch_texts, truncation=True, padding=True, 
                    max_length=MAX_LENGTH, return_tensors='pt'
                ).to(self.device)

                optimizer.zero_grad()
                outputs = self.model(**encodings, labels=batch_labels)
                loss = outputs.loss
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / (len(df) / batch_size)
            print(f"  Средняя потеря: {avg_loss:.4f} | Время: {time.time()-start_time:.1f} сек")

        self.save_model()
        print("✅ Обучение успешно завершено!")

    def save_model(self):
        save_path = MODELS_DIR / "bert_classifier"
        save_path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        print(f"Модель сохранена в: {save_path}")

    def load_model(self):
        model_path = MODELS_DIR / "bert_classifier"
        if model_path.exists():
            self.tokenizer = BertTokenizer.from_pretrained(model_path)
            self.model = BertForSequenceClassification.from_pretrained(model_path)
            self.model.to(self.device)
            self.model.eval()
            print("✅ Модель успешно загружена")
            return True
        return False

    def predict(self, text: str):
        if not self.load_model():
            return "Модель не загружена", 0.0

        cleaned, lemmatized = preprocess_appeal(text)

        inputs = self.tokenizer(lemmatized, truncation=True, padding=True,
                              max_length=MAX_LENGTH, return_tensors='pt').to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=1)
            pred_idx = torch.argmax(probs, dim=1).item()
            confidence = probs[0][pred_idx].item()

        inverse = joblib.load(MODELS_DIR / 'inverse_label_encoder.pkl')
        category = inverse.get(pred_idx, "Неизвестно")

        return category, confidence


# ====================== ЗАПУСК ОБУЧЕНИЯ ======================
if __name__ == "__main__":
    print("🚀 Запуск обучения модели...")

    data_path = Path("data/escalated_appeals.xlsx")
    if data_path.exists():
        df = pd.read_excel(data_path)
        print(f"Загружено {len(df)} обращений")
    else:
        print("Датасет не найден!")
        sys.exit(1)

    classifier = BertClassifier(num_labels=len(df['category'].unique()))
    classifier.train(df, epochs=3, batch_size=12)   # epochs=2 для быстрого теста