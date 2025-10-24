# Imagen base moderna con Python 3.13
FROM python:3.13-slim AS base

# Variables de entorno recomendadas
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV UV_SYSTEM_PYTHON=1

WORKDIR /app

# Instalar dependencias del sistema necesarias para Pillow y compilación
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libjpeg-dev \
    zlib1g-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/* \
    && pip cache purge \
    && rm -rf /root/.cache/huggingface /root/.cache/torch

# -------------------------------------------------------------------
# Etapa 1: instalar uv y dependencias de proyecto
# -------------------------------------------------------------------
FROM base AS builder

# Instalar uv (gestor de dependencias ultrarrápido)
RUN pip install --no-cache-dir uv

# Copiar archivo de dependencias
COPY pyproject.toml .
# Copiar lock file si existe
COPY uv.lock* .

# Instalar dependencias del proyecto en el entorno del sistema
RUN uv pip install --system --no-cache-dir .

# -------------------------------------------------------------------
# Etapa 2: copiar el código fuente y preparar entorno de ejecución
# -------------------------------------------------------------------
FROM base AS runtime

# Copiar dependencias ya instaladas
COPY --from=builder /usr/local /usr/local

# Copiar el código fuente
COPY api ./api
COPY templates ./templates

# Crear carpeta para la librería mylib
RUN mkdir -p /app/mylib
# Copiar archivos esenciales de la librería mylib
COPY mylib/model.py mylib/__init__.py /app/mylib/


# Crear carpeta de trabajo para imágenes o archivos
RUN mkdir -p /app/uploads

# Exponer puerto de la API FastAPI
EXPOSE 8000

# Comando por defecto: iniciar FastAPI con uvicorn
CMD ["uvicorn", "api.fastapi_app:app", "--host", "0.0.0.0", "--port", "8000"]

