FROM python:3.10-slim

# Installer Git LFS et dépendances système
RUN apt-get update && \
    apt-get install -y git git-lfs && \
    git lfs install && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copier et installer les dépendances
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copier tous les fichiers (y compris best_model.pth via LFS)
COPY . .

# Port exposé (Railway utilise la variable PORT)
EXPOSE 8000

# Lancement avec PORT dynamique
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000}"]
