import os
import requests as rq
import streamlit as st
from datetime import datetime


URL = f"http://backend:{os.getenv('BACKEND_PORT')}/"
# URL = "http://localhost:8502"

st.set_page_config(page_title="GAGARIN HACK 2024", layout="wide")

# st.markdown("""
#     <style>
#     [data-testid="stHeader"] {
#       background-color: #020e42;
#       color: white;
#     }
#
#     [data-testid="stAppViewContainer"] {
#       background: linear-gradient(to bottom, #020e42, #262730);
#       color: white;
#     }
#     </style>
# """, unsafe_allow_html=True)

st.title("Punk Butterfly - Gagarin Hack 2024", anchor=False)
st.header("")

uploaded_file = st.file_uploader('Скан автомобильных документов (ВУ, СТС, ПТС)', accept_multiple_files=False)

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

# curr_time = datetime.now()
# try:
#     if headers is not None and source_img is not None and image_bin is not None:
#         db_request = {
#             "ipv4": "0.0.0.0",
#             "time": str(curr_time),
#
#             "source_img_bindata": str(source_img),
#             "source_img_filename": f"{curr_time}.png",
#
#             "proceed_img_bindata": str(image_bin),
#             "proceed_img_filename": f"proceed_{curr_time}.png",
#
#             "recog_type": headers['type'],
#             "confidence": headers['confidence'],
#             "series": headers['series'],
#             "number":  headers['number'],
#             "page_number": headers['page_number'],
#         }
#
#         # request_to_db = rq.post(f"http://158.160.17.229:8503/log", json=db_request)
#
#         print(request_to_db)
# except:
#     print("отправка на дб не удалfсь")

