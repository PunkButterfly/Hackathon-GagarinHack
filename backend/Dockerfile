FROM python:3.12
WORKDIR /backend

COPY . /backend/

RUN pip install -r requirements.txt
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

CMD python api.py
