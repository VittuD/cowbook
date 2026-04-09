FROM ultralytics/ultralytics:8.4.34

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

COPY pyproject.toml readme.md config.json /app/
COPY src /app/src

RUN python -m pip install --upgrade pip && \
    python -m pip install -e .

COPY assets /app/assets
COPY configs /app/configs
COPY models /app/models
COPY sample_data /app/sample_data

ENTRYPOINT ["python", "-m", "cowbook"]
CMD ["--config", "config.json"]
