FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    OPENBLAS_CORETYPE=ARMV8 \
    OMP_NUM_THREADS=1

RUN pip install --no-cache-dir torch==2.3.1 pytest==8.4.1

RUN mkdir -p /solution /logs/verifier /paper
COPY Forward_Projection_Paper.pdf /paper/Forward_Projection_Paper.pdf

WORKDIR /solution
