import requests as rq
import streamlit as st

# Внутрення сеть докер композа
st.write(rq.get("http://backend:8502/").text)



