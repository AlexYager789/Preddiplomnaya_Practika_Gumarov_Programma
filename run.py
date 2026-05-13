# run.py
import os
import sys
from pathlib import Path

print("🚀 Запуск системы интеллектуального анализа обращений...\n")

# Генерация данных, если нет
if not Path("data/sample_data.xlsx").exists():
    print("📊 Генерация тестовых данных...")
    os.system("python data/generate_sample_data.py")

# Обучение модели, если не обучена
if not Path("models/saved/bert_classifier").exists():
    print("🧠 Обучение модели (это может занять несколько минут)...")
    os.system("python models/bert_classifier.py")

print("🌐 Запуск веб-интерфейса...")
os.system("streamlit run main.py")