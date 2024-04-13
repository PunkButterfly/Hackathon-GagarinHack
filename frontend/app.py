import requests as rq
import streamlit as st

import shutil


# Внутрення сеть докер композа
# st.write(rq.get("http://backend:8502/").text)

URL = "http://backend:8502"
# URL = "http://localhost:8502"

st.set_page_config(page_title="GAGARIN HACK 2024", layout="wide")

uploaded_file = st.file_uploader('Скан документа', accept_multiple_files=False)
# print(uploaded_file.read())
# uploaded_file_bytes = open(uploaded_file, 'rb')
# uploaded_file_name = uploaded_file.name

response = None

if uploaded_file:
    response = rq.post(f"{URL}/detect_punk_client/", files={"file": uploaded_file.read()}).json()

if response:
    features_response = response["proceed_image_name"]

    img_response = rq.post(f"{URL}/get_image_by_path/", params={"img_path": features_response})

    proceed_img_path = features_response.split('/')[-1]
    if img_response.status_code == 200:
        with open(features_response.split('/')[-1], 'wb') as out_file:
            out_file.write(img_response.content)

image, proceed_img, response_body = st.columns(3, gap='small')

with image:
    if uploaded_file:
        st.image(uploaded_file, caption="Загруженный снимок документа")

with proceed_img:
    if uploaded_file:
        st.image(proceed_img_path, caption="Выделенные данные на снимке")

with response_body:
    if uploaded_file:
        st.write(f"Тип документа (type): {response['type']}")
        st.write(f"Уверенность в предсказании (confidence): {response['confidence']}")
        st.write(f"Серия документа (series): {response['series']}")
        st.write(f"Номер страницы (page_number): {response['page_number']}")
        st.write(f"Номер документа (number): {response['number']}")

        st.write(f"Распознанный текст: {response['recognited_text']}")






