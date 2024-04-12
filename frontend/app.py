import io
import json
from fastapi import UploadFile
import requests as rq
import streamlit as st

# Внутрення сеть докер композа
# st.write(rq.get("http://backend:8502/").text)

URL = "http://158.160.17.229:8502"

st.set_page_config(page_title="GAGARIN HACK 2024", layout="wide")

uploaded_file = st.file_uploader('Скан документа', accept_multiple_files=False)
uploaded_file_bytes = uploaded_file.read()
uploaded_file_name = uploaded_file.name
uploaded_file_byteio = io.BytesIO(uploaded_file_bytes)

posted_file = UploadFile(filename=uploaded_file_name, file=uploaded_file_byteio)

response = rq.post(f"{URL}/process_image/", files={"file": posted_file}).json()
#
st.write(response)



