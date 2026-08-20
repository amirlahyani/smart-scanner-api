FROM python:3.10-slim

# Installer Git LFS
RUN apt-get update && apt-get install -y git git-lfs && git lfs install

WORKDIR /app

# Copier et installer les dépendances
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copier le reste (y compris le .pth via LFS)
COPY . .

# Nettoyer
RUN apt-get clean

CMD ["python", "app.py"]
