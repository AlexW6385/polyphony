FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app

COPY pyproject.toml README.md ./
COPY polyphony ./polyphony
RUN pip install .

COPY voices ./voices
COPY senses.yaml ./
COPY static ./static
COPY scripts ./scripts

EXPOSE 8000
CMD ["python", "-m", "polyphony.main"]
