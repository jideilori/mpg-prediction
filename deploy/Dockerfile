FROM tiangolo/uvicorn-gunicorn-fastapi:python3.8 as base

ADD . ./mpg-prediction

ADD requirements.txt ./mpg-prediction

ENV PYTHONPATH $PYTHONPATH:./mpg-prediction

WORKDIR ./mpg-prediction

RUN pip3 install --no-cache-dir -r requirements.txt

# COPY ./mpg_prediction /mpg_prediction

# WORKDIR /mpg_prediction

# RUN pip3 install --no-cache-dir -r requirements.txt
