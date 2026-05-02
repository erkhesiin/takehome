FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV ATEN_CPU_CAPABILITY=default
ENV OMP_NUM_THREADS=1
ENV MKL_NUM_THREADS=1
ENV OPENBLAS_NUM_THREADS=1
ENV NUMEXPR_NUM_THREADS=1

WORKDIR /app

COPY DTR_Paper.pdf /app/DTR_Paper.pdf

RUN pip install --no-cache-dir pytest==8.4.1 \
    && pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.3.1
