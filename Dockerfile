FROM python:3.12-slim

WORKDIR /app_06_deploy

RUN pip install uv

COPY pyproject.toml uv.lock ./

RUN uv sync --locked

COPY . .

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "02_rag_config_errores:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD ["python3", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"]