# main.py
import streamlit as st
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent))

from config import PAGE_TITLE, PAGE_ICON
from database.session import init_database

# Инициализация БД при первом запуске
if "db_initialized" not in st.session_state:
    init_database()
    st.session_state.db_initialized = True

# Настройка страницы
st.set_page_config(
    page_title=PAGE_TITLE,
    page_icon=PAGE_ICON,
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📞 Интеллектуальный анализ обращений call-центра")
st.markdown("**ЧОУ ВО «Московский университет им. С.Ю. Витте»**")

# Боковая панель
with st.sidebar:
    st.image("https://via.placeholder.com/180x80?text=МУИВ", width=180)
    st.title("Меню")

    page = st.radio(
        "Перейти к разделу:",
        ["🏠 Главная", 
         "🔍 Анализ обращений", 
         "📊 История", 
         "📈 Кластеризация", 
         "📋 Отчёты", 
         "⚙️ Администрирование"]
    )

    st.divider()
    st.caption("**Гумаров Рафаэль Ленарович**\nПреддипломная практика — 2026")

# Импорт страниц
if page == "🏠 Главная":
    st.success("Система готова к работе!")
    col1, col2, col3 = st.columns(3)
    with col1: st.metric("Всего обращений", "1 247", "↑24")
    with col2: st.metric("Точность модели", "81.7%", "↑2.3")
    with col3: st.metric("Эскалировано", "18%", "↓4")

elif page == "🔍 Анализ обращений":
    from gui.pages._1_Анализ_обращений import analysis_page
    analysis_page()

elif page == "📊 История":
    st.header("📊 История обращений")
    st.info("Здесь будет таблица всех обработанных обращений (следующий этап)")

elif page == "📈 Кластеризация":
    st.header("📈 Кластеризация системных проблем")
    st.info("Модуль кластеризации в разработке")

elif page == "📋 Отчёты":
    st.header("📋 Генерация отчётов")
    st.info("Экспорт в Excel / Word — в разработке")

elif page == "⚙️ Администрирование":
    st.header("⚙️ Панель администратора")
    st.warning("Доступ только для роли Admin")

st.caption("© 2026 Гумаров Рафаэль Ленарович | Версия 0.2")