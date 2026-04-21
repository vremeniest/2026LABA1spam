FROM python:3.9-slim

WORKDIR /app

# Установка зависимостей системы
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY config.ini .
COPY data/ ./data/

# Создаём пустую папку для модели (она создастся при обучении)
RUN mkdir -p models

EXPOSE 8000

# Команда для запуска: сначала обучаем модель, потом API
CMD python src/train.py && python src/api.py