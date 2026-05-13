
import re
import string
from typing import List, Tuple
import pymorphy3
from natasha import Segmenter, MorphVocab, NewsEmbedding, NewsMorphTagger, Doc

# Инициализация
morph = pymorphy3.MorphAnalyzer()
segmenter = Segmenter()
emb = NewsEmbedding()
morph_tagger = NewsMorphTagger(emb)
morph_vocab = MorphVocab()


def clean_text(text: str) -> str:
    """Базовая очистка текста"""
    if not text:
        return ""
    
    # Удаляем HTML-теги
    text = re.sub(r'<.*?>', '', text)
    # Удаляем лишние пробелы
    text = re.sub(r'\s+', ' ', text)
    # Удаляем специальные символы
    text = re.sub(r'[^\w\s\.,!?-]', '', text)
    return text.strip()


def remove_personal_data(text: str) -> str:
    """Удаление персональных данных (упрощённо)"""
    # Телефоны
    text = re.sub(r'(\+7|8)?[\s-]?\(?\d{3}\)?[\s-]?\d{3}[\s-]?\d{2}[\s-]?\d{2}', '[ТЕЛЕФОН]', text)
    # Email
    text = re.sub(r'[\w\.-]+@[\w\.-]+', '[EMAIL]', text)
    # ФИО (простая эвристика)
    text = re.sub(r'\b[А-ЯЁ][а-яё]+\s+[А-ЯЁ][а-яё]+\s+[А-ЯЁ][а-яё]+\b', '[ФИО]', text)
    return text


def lemmatize_text(text: str) -> str:
    """Лемматизация текста"""
    doc = Doc(text)
    doc.segment(segmenter)
    doc.tag_morph(morph_tagger)
    
    lemmas = []
    for token in doc.tokens:
        token.lemmatize(morph_vocab)
        if token.lemma and token.lemma not in string.punctuation:
            lemmas.append(token.lemma.lower())
    
    return ' '.join(lemmas)


def preprocess_appeal(text: str) -> Tuple[str, str]:
    """
    Полная предобработка обращения
    Возвращает: (очищенный_текст, лемматизированный_текст)
    """
    original = text
    text = clean_text(text)
    text = remove_personal_data(text)
    lemmatized = lemmatize_text(text)
    
    return text, lemmatized


# Для тестирования
if __name__ == "__main__":
    test_text = "Здравствуйте, у меня проблема с расписанием у преподавателя Ивановой Ирины Петровны, телефон +79161234567"
    clean, lemma = preprocess_appeal(test_text)
    print("Очищенный:", clean)
    print("Лемматизированный:", lemma)