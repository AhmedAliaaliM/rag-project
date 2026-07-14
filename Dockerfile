FROM python:3.11-slim
WORKDIR /app
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir --timeout=300 -r requirements.txt
COPY src/ ./src/
COPY data/eval_set.json ./data/
COPY data/chunks/ ./data/chunks/
RUN mkdir -p logs data/uploads
EXPOSE 8000
CMD ["python", "src/api.py"]
