FROM python:3.8-slim

WORKDIR /mpg-prediction

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

COPY ./requirements.txt .

RUN pip install -r requirements.txt

COPY ./mpg_prediction /mpg-prediction/mpg_prediction

CMD [ "python", "mpg_prediction/main.py"] 
