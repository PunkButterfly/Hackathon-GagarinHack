FROM python:3.12
WORKDIR /container

COPY . /container/

RUN pip install -r requirements.txt

ENV PORT 8501
EXPOSE $PORT

CMD streamlit run app.py --server.port $PORT

