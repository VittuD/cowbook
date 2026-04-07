FROM ultralytics/ultralytics:8.4.34

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

COPY . /app

RUN python -m pip install --upgrade pip && \
    python -m pip install -e .

ENTRYPOINT ["python", "-m", "cowbook"]
CMD ["--config", "config.json"]
