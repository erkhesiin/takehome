FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY DTR_Paper.pdf /app/DTR_Paper.pdf

RUN pip install --no-cache-dir pytest==8.4.1 \
    && pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.3.1
