FROM python:3.13-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV DATABASE_BACKEND=local
ENV DATABASE_URL=postgresql://weather_user:weather_password@postgres:5432/weather_rag
ENV OLLAMA_HOST=http://host.docker.internal:11434
ENV WEATHER_LLM_MODEL=llama3.2:1b
ENV FLASK_RUN_HOST=0.0.0.0
ENV FLASK_RUN_PORT=8000

EXPOSE 8000

CMD ["python", "app.py"]
