# data/generate_escalated_dataset.py
import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from utils.text_preprocessing import preprocess_appeal

def generate_escalated_dataset(n_samples=500):
    """Генерация датасета с акцентом на эскалируемые обращения"""
    np.random.seed(42)
    
    categories = [
        "Расписание", "Оплата и финансы", "Документы и справки",
        "Технические проблемы", "Преподаватели и качество обучения",
        "Поступление и приёмная комиссия", "Общежитие и проживание", "Административные вопросы"
    ]
    
    # Более реалистичные шаблоны эскалируемых обращений
    escalated_templates = {
        "Расписание": [
            "Расписание постоянно меняется, уже третий раз переносят пару. Группа в шоке.",
            "Преподаватель не выходит на связь, пара по расписанию, а его нет уже 40 минут.",
            "В расписании ошибка — два важных экзамена в один день."
        ],
        "Оплата и финансы": [
            "Оплата не прошла уже третий раз, хотя деньги списались.",
            "Не возвращают переплату за прошлый семестр уже 2 месяца.",
            "Повысили стоимость обучения без предупреждения."
        ],
        "Преподаватели и качество обучения": [
            "Преподаватель ведёт занятия очень плохо, ничего не объясняет.",
            "Преподаватель оскорбляет студентов на занятиях.",
            "Не принимает лабораторные работы без причины."
        ],
        # ... и т.д.
    }
    
    data = []
    
    for i in range(n_samples):
        category = np.random.choice(categories)
        is_escalated = np.random.choice([True, False], p=[0.35, 0.65])  # 35% эскалируемых
        
        if is_escalated and category in escalated_templates:
            base_text = np.random.choice(escalated_templates[category])
        else:
            base_text = f"Проблема по направлению '{category}'. Прошу помочь разобраться."
        
        # Добавляем эмоциональность для эскалируемых
        if is_escalated:
            base_text += " " + np.random.choice([
                "Это уже не первый раз, ситуация критичная.",
                "Требую разобраться на более высоком уровне.",
                "Я уже писал несколько раз, реакции нет.",
                "Это сильно влияет на мою успеваемость."
            ])
        
        cleaned, lemmatized = preprocess_appeal(base_text)
        
        data.append({
            'id': i + 1,
            'original_text': base_text,
            'appeal_text': cleaned,
            'lemmatized': lemmatized,
            'category': category,
            'is_escalated': is_escalated,
            'sentiment': 'negative' if is_escalated else np.random.choice(['negative', 'neutral']),
            'urgency': 'high' if is_escalated else np.random.choice(['high', 'medium', 'low']),
            'escalation_reason': 'Многократные обращения' if is_escalated else None
        })
    
    df = pd.DataFrame(data)
    return df


if __name__ == "__main__":
    print("Генерация датасета с эскалируемыми обращениями...")
    df = generate_escalated_dataset(600)
    
    df.to_excel("data/escalated_appeals.xlsx", index=False)
    df.to_csv("data/escalated_appeals.csv", index=False)
    
    print(f"✅ Датасет успешно создан!")
    print(f"   Всего обращений: {len(df)}")
    print(f"   Из них эскалировано: {df['is_escalated'].sum()} ({df['is_escalated'].mean():.1%})")
    print(f"   Файлы сохранены в папке data/")
    
    print("\nПример эскалируемого обращения:")
    print(df[df['is_escalated']].sample(1)['original_text'].iloc[0])