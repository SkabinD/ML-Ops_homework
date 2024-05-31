import streamlit as st
import pandas as pd
import app.src.backend as backend

st.header("МТS ML-Ops Homework")

st.write("#### Описание")
st.write("Моя реализация решения домашнего задания по ML-Ops МТС ШАД :rocket:")
st.write("Для работы необходимо загрузить файл с тестовыми данными в блок \"Загрузка файла\"")
st.write("Приложение обработает загруженные данные и выдаст .csv-файл с предсказаниями, график плотности распределения предсказаний, а также наиболее важные признаки")

st.write("#### Функционал")

c1, c4 = st.columns(2)
c3, c2 = st.columns(2)

def plot_pred_proba(plot, item):
    with item:
        st.pyplot(plot)
        with open("app/output/predict_density_distrib.png", "rb") as f:
            st.download_button(label="Скачать файл", data=f.read(), file_name="PDF.png", mime="image/png")

def show_dataframe(submission, item):
    with item:
        st.dataframe(submission, height=300, use_container_width=True)
        with open("app/output/submission.csv", "r") as f:
            st.download_button(label="Скачать файл",  file_name="submission.csv", data=f.read(), mime="text/csv")

def get_imp_features(feature_importance, item):
    with item:
        st.json(feature_importance)
        with open("app/output/features_importance.json", "r", encoding='utf-8') as f:
            st.download_button(label="Скачать файл",  file_name="feature_importance.json", data=f.read(), mime="application/json")

def frontend_routine(submission, plot, feature_importance):
    plot_pred_proba(plot, c3)
    show_dataframe(submission, c2)
    get_imp_features(feature_importance, c4)

with st.container():
    c2.write("###### Сохранение файла")

with st.container():
    c3.write("###### Плотность распределения предсказаний")
    c4.write("###### Наиболее важные признаки")

with c1:
    upload = st.file_uploader(label="###### Загрузка файла", type=["csv"])
    if upload:
        submission, plot, feature_importance = backend.predict_routine(upload)
        frontend_routine(submission, plot, feature_importance)
    else:
        print("Add something in idle status")
    