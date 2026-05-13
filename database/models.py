from sqlalchemy import Column, Integer, String, Text, DateTime, Float, ForeignKey, Boolean
from sqlalchemy.orm import declarative_base, relationship
from datetime import datetime

Base = declarative_base()


class User(Base):
    """Пользователи системы"""
    __tablename__ = 'users'

    id = Column(Integer, primary_key=True)
    username = Column(String(50), unique=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    full_name = Column(String(100))
    role = Column(String(20), nullable=False)  # admin, analyst, operator
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f"<User {self.username} ({self.role})>"


class Appeal(Base):
    """Обращение из call-центра"""
    __tablename__ = 'appeals'

    id = Column(Integer, primary_key=True)
    appeal_text = Column(Text, nullable=False)
    original_text = Column(Text)  # исходный текст до обработки
    category = Column(String(50))                    # предсказанная категория
    true_category = Column(String(50))               # реальная категория (для обучения)
    sentiment = Column(String(20))                   # positive, neutral, negative
    urgency = Column(String(20))                     # low, medium, high
    confidence = Column(Float)                       # уверенность модели
    is_escalated = Column(Boolean, default=False)    # было ли эскалировано
    created_at = Column(DateTime, default=datetime.utcnow)
    processed_at = Column(DateTime)
    user_id = Column(Integer, ForeignKey('users.id'))

    # Связь с предсказаниями
    predictions = relationship("Prediction", back_populates="appeal")


class Prediction(Base):
    """Результаты работы модели"""
    __tablename__ = 'predictions'

    id = Column(Integer, primary_key=True)
    appeal_id = Column(Integer, ForeignKey('appeals.id'))
    model_version = Column(String(50))
    predicted_category = Column(String(50))
    predicted_sentiment = Column(String(20))
    predicted_urgency = Column(String(20))
    confidence_score = Column(Float)
    processing_time = Column(Float)  # в секундах
    created_at = Column(DateTime, default=datetime.utcnow)

    appeal = relationship("Appeal", back_populates="predictions")


class Cluster(Base):
    """Кластеры обращений (системные проблемы)"""
    __tablename__ = 'clusters'

    id = Column(Integer, primary_key=True)
    cluster_id = Column(Integer)
    cluster_name = Column(String(100))
    description = Column(Text)
    size = Column(Integer)
    main_category = Column(String(50))
    created_at = Column(DateTime, default=datetime.utcnow)


# Функция инициализации БД
def init_db(engine):
    Base.metadata.create_all(engine)