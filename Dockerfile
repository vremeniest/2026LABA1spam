FROM python:3.9-slim

WORKDIR /app

# Установка зависимостей системы для OpenCV
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY models/ ./models/
COPY config.ini .

EXPOSE 8000

CMD ["python", "src/api.py"]