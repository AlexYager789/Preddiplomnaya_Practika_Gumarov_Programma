# database/session.py
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from config import BASE_DIR

# Создаём SQLite базу
DATABASE_URL = f"sqlite:///{BASE_DIR}/call_center.db"
engine = create_engine(DATABASE_URL, echo=False)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db():
    """Получить сессию БД"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_database():
    """Инициализация базы данных"""
    from database.models import Base, init_db
    init_db(engine)
    print("База данных успешно инициализирована!")