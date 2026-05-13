# gui/pages/1_Анализ_обращений.py
import streamlit as st
import pandas as pd
from pathlib import Path
import sys
import time

sys.path.append(str(Path(__file__).parent.parent.parent))

from models.bert_classifier import BertClassifier
from utils.text_preprocessing import preprocess_appeal
from config import CATEGORIES


def analysis_page():
    st.header("🔍 Анализ нового обращения")
    st.markdown("Загрузите обращение или введите текст вручную")

    tab1, tab2 = st.tabs(["📝 Ввести текст", "📁 Загрузить файл"])

    with tab1:
        appeal_text = st.text_area(
            "Введите текст обращения:",
            height=150,
            placeholder="Здравствуйте, у меня проблема с расписанием..."
        )

        if st.button("🔎 Проанализировать обращение", type="primary"):
            if appeal_text.strip():
                with st.spinner("Модель анализирует обращение..."):
                    start_time = time.time()
                    
                    classifier = BertClassifier()
                    if not classifier.load_model():
                        st.error("Модель не найдена. Сначала обучите модель.")
                        return
                    
                    category, confidence = classifier.predict(appeal_text)
                    processing_time = time.time() - start_time

                    # Результаты
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Категория", category)
                    with col2:
                        st.metric("Уверенность модели", f"{confidence:.1%}")
                    with col3:
                        st.metric("Время анализа", f"{processing_time:.2f} сек")

                    # Дополнительная информация
                    st.subheader("Подробный результат")
                    cleaned, lemmatized = preprocess_appeal(appeal_text)
                    
                    st.info(f"**Очищенный текст:** {cleaned}")
                    st.caption(f"**Лемматизированный:** {lemmatized}")

                    # Рекомендации
                    st.subheader("Рекомендации")
                    if confidence > 0.85:
                        st.success("Высокая уверенность модели. Можно перенаправить в соответствующее подразделение.")
                    elif confidence > 0.6:
                        st.warning("Средняя уверенность. Рекомендуется проверить вручную.")
                    else:
                        st.error("Низкая уверенность. Требуется эскалация.")

            else:
                st.warning("Введите текст обращения!")

    with tab2:
        uploaded_file = st.file_uploader("Загрузите файл с обращениями (Excel или CSV)", 
                                       type=['xlsx', 'csv'])
        
        if uploaded_file is not None:
            if st.button("Обработать файл"):
                with st.spinner("Обработка файла..."):
                    try:
                        if uploaded_file.name.endswith('.csv'):
                            df = pd.read_csv(uploaded_file)
                        else:
                            df = pd.read_excel(uploaded_file)

                        st.success(f"Загружено {len(df)} обращений")
                        st.dataframe(df.head(5))

                        # Здесь можно добавить массовую обработку
                        if st.button("Запустить анализ всех обращений"):
                            st.info("Массовый анализ будет реализован на следующем этапе.")

                    except Exception as e:
                        st.error(f"Ошибка при чтении файла: {e}")


# Для тестирования страницы
if __name__ == "__main__":
    analysis_page()