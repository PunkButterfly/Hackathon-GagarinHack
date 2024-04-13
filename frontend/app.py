import io
import json
from fastapi import UploadFile
import requests as rq
import streamlit as st

from datetime import datetime

import shutil


# Внутрення сеть докер композа
# st.write(rq.get("http://backend:8502/").text)

URL = "http://backend:8502"
# URL = "http://localhost:8502"

st.set_page_config(page_title="GAGARIN HACK 2024", layout="wide")

uploaded_file = st.file_uploader('Скан документа', accept_multiple_files=False)

response = None

if uploaded_file:
    source_img = uploaded_file.read()
    response = rq.post(f"{URL}/detect/", files={"file": source_img})

if response:
    image_bin = response.content
    headers = response.headers
    

image, proceed_img, response_body = st.columns(3, gap='small')

with image:
    if uploaded_file:
        st.image(uploaded_file, caption="Загруженный снимок документа")

with proceed_img:
    if uploaded_file:
        st.image(image_bin, caption="Выделенные данные на снимке")

with response_body:
    if uploaded_file:
        st.write(f"Тип документа (type): {headers['type']}")
        st.write(f"Уверенность в предсказании (confidence): {headers['confidence']}")
        st.write(f"Номер документа (number): {headers['number']}")
        st.write(f"Серия документа (series): {headers['series']}")
        st.write(f"Номер страницы (page_number): {headers['page_number']}")

curr_time = datetime.now()
try:
    if headers is not None and source_img is not None and image_bin is not None:
        db_request = {
            "ipv4": "0.0.0.0",
            "time": curr_time,

            # "source_img_bindata": source_img,
            "source_img_bindata": "TEST",
            "source_img_filename": f"{curr_time}.png",

            # "proceed_img_bindata": img_response.content,
            "proceed_img_bindata": "TEST",
            "proceed_img_filename": f"proceed_{curr_time}.png",

            "recog_type": response['type'],
            "confidence": response['confidence'],
            "series": response['series'],
            "number":  response['number'],
            "page_number": response['page_number'],
            "recog_text": response['recognited_text']
        }

        request_to_db = rq.post(f"http://158.160.17.229:8503/log/", data=db_request)

        print(request_to_db)
except:
    print("отправка на дб не удалсь")

