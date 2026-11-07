# Imagen de la API de inferencia. Dependencias fijadas en requirements.txt (incluye mlflow==3.18.0,
# la misma versión que el servidor de tracking, para que el modelo cargue sin discrepancias).
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 MLFLOW_DISABLE_AGENT_HINT=1
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/

# Usuario sin privilegios
RUN useradd --create-home appuser
USER appuser

EXPOSE 8000
ENV MLFLOW_TRACKING_URI=http://mlflow:5000 MODEL_NAME=desercion_caso_x MODEL_ALIAS=champion
CMD ["uvicorn", "src.serve.api:app", "--host", "0.0.0.0", "--port", "8000"]
