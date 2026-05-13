
import pandas as pd
import numpy as np
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from utils.text_preprocessing import preprocess_appeal

def generate_sample_data(n_samples=300):
    """Генерация синтетического датасета"""
    np.random.seed(42)
    
    categories = ["Расписание", "Оплата и финансы", "Документы и справки", 
                 "Технические проблемы", "Преподаватели", "Поступление", 
                 "Общежитие", "Другое"]
    
    sentiments = ["negative", "neutral", "positive"]
    urgencies = ["high", "medium", "low"]
    
    data = []
    
    templates = {
        "Расписание": ["Не могу найти расписание на следующую неделю", "Расписание сдвинулось, занятия отменили", "Когда будет экзамен по ..."],
        "Оплата и финансы": ["Не прошла оплата за обучение", "Когда вернут переплату", "Как оплатить общежитие"],
        "Документы и справки": ["Нужна справка 2-НДФЛ", "Не могу получить академическую справку"],
        # ... остальные категории
    }
    
    for i in range(n_samples):
        category = np.random.choice(categories)
        sentiment = np.random.choice(sentiments, p=[0.5, 0.35, 0.15])
        urgency = np.random.choice(urgencies, p=[0.3, 0.5, 0.2])
        is_escalated = np.random.choice([True, False], p=[0.25, 0.75])
        
        # Генерация текста
        base_text = f"Здравствуйте. У меня {category.lower()}. {np.random.choice(templates.get(category, ['Проблема']))}."
        if sentiment == "negative":
            base_text += " Это очень срочно и неудобно!"
        
        cleaned, lemmatized = preprocess_appeal(base_text)
        
        data.append({
            'id': i+1,
            'appeal_text': cleaned,
            'original_text': base_text,
            'category': category,
            'sentiment': sentiment,
            'urgency': urgency,
            'is_escalated': is_escalated,
            'lemmatized': lemmatized
        })
    
    df = pd.DataFrame(data)
    return df


if __name__ == "__main__":
    df = generate_sample_data(350)
    df.to_excel("data/sample_data.xlsx", index=False)
    df.to_csv("data/sample_data.csv", index=False)
    print(f"Создано {len(df)} записей. Файл сохранён: data/sample_data.xlsx")