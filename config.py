


### 3. `config.py`


# config.py
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Пути
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models" / "saved"
REPORTS_DIR = BASE_DIR / "reports"
LOGS_DIR = BASE_DIR / "logs"

# Создаём директории
for dir_path in [DATA_DIR, MODELS_DIR, REPORTS_DIR, LOGS_DIR, DATA_DIR/"raw", DATA_DIR/"processed"]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Настройки модели
MODEL_NAME = "ai-forever/ruBert-base"  # или "DeepPavlov/rubert-base-cased"
MAX_LENGTH = 512
BATCH_SIZE = 16

# Настройки Streamlit
PAGE_TITLE = "Интеллектуальный анализ обращений call-центра"
PAGE_ICON = "📞"

# Роли пользователей
ROLES = {
    "admin": "Администратор",
    "analyst": "Аналитик",
    "operator": "Оператор"
}

# Категории обращений (будут расширены)
CATEGORIES = [
    "Расписание", "Оплата и финансы", "Документы и справки",
    "Технические проблемы", "Преподаватели", "Поступление",
    "Общежитие", "Другое"
]