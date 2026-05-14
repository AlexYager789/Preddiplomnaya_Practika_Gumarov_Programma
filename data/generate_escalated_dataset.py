# data/generate_escalated_dataset.py
import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from utils.text_preprocessing import preprocess_appeal


def generate_escalated_dataset(n_samples=600):
    """Создание датасета с акцентом на эскалируемые обращения"""
    np.random.seed(42)
    
    categories = [
        "Расписание", "Оплата и финансы", "Документы и справки",
        "Технические проблемы", "Преподаватели и качество обучения",
        "Поступление и приёмная комиссия", "Общежитие и проживание", 
        "Административные вопросы", "Другое"
    ]
    
    data = []
    
    for i in range(n_samples):
        category = np.random.choice(categories)
        is_escalated = np.random.rand() < 0.35  # 35% эскалируемых
        
        if is_escalated:
            text = np.random.choice([
                f"Уже третий раз не могу решить вопрос по {category.lower()}. Прошу разобраться на высшем уровне.",
                f"Ситуация критичная. По {category.lower()} ничего не решается уже долгое время.",
                f"Обращался多次, но проблему по {category.lower()} так и не решили.",
                f"Это недопустимо! Требую реакции руководства по вопросу {category.lower()}."
            ])
        else:
            text = f"Здравствуйте, у меня вопрос по {category.lower()}."
        
        cleaned, lemmatized = preprocess_appeal(text)
        
        data.append({
            'id': i + 1,
            'original_text': text,
            'appeal_text': cleaned,
            'lemmatized': lemmatized,
            'category': category,
            'is_escalated': is_escalated,
            'sentiment': 'negative' if is_escalated else 'neutral',
            'urgency': 'high' if is_escalated else 'medium'
        })
    
    df = pd.DataFrame(data)
    return df


if __name__ == "__main__":
    print("Генерация датасета эскалируемых обращений...")
    df = generate_escalated_dataset(600)
    
    output_dir = Path("data")
    output_dir.mkdir(exist_ok=True)
    
    df.to_excel(output_dir / "escalated_appeals.xlsx", index=False)
    df.to_csv(output_dir / "escalated_appeals.csv", index=False)
    
    print(f"✅ Датасет успешно создан! ({len(df)} записей)")
    print(f"   Эскалируемых обращений: {df['is_escalated'].sum()} ({df['is_escalated'].mean():.1%})")