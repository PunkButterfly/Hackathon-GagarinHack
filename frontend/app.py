import requests as rq
import streamlit as st

# Внутрення сеть докер композа
# st.write(rq.get("http://backend:8502/").text)

URL = "http://backend:8502"

st.set_page_config(page_title="GAGARIN HACK 2024", layout="wide")

uploaded_file = st.file_uploader('Скан документа', accept_multiple_files=False)
# print(uploaded_file.read())
# uploaded_file_bytes = open(uploaded_file, 'rb')
# uploaded_file_name = uploaded_file.name

response = rq.post(f"{URL}/detect/", files={"file": uploaded_file}).json()
# #
st.write(response)



