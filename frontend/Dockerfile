ARG FRONTEND_PORT

FROM python:3.12
WORKDIR /container

COPY . /container/

RUN pip install -r requirements.txt

EXPOSE $FRONTEND_PORT

CMD streamlit run app.py --server.port $FRONTEND_PORT

