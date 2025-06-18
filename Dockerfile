# Dockerfile dla projektu Vision Transformer vs CNN
FROM python:3.9-slim

# Ustawienia systemowe
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Zainstaluj zależności systemowe
RUN apt-get update && apt-get install -y \
    git \
    curl \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Ustawienia robocze
WORKDIR /app

# Skopiuj requirements i zainstaluj zależności Python
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Skopiuj kod projektu
COPY . .

# Utwórz niezbędne foldery
RUN mkdir -p data results logs checkpoints

# Ustaw uprawnienia
RUN chmod +x test_setup.py main.py

# Domyślna komenda - test konfiguracji
CMD ["python", "test_setup.py"] 