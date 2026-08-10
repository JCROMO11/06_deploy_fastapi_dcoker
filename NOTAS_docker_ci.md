# Notas Niveles 3-5 (Docker + CI + Deploy)

Estos niveles producen archivos de infraestructura, no solo .py. Crea aquí:

- `Dockerfile` (3.1: multi-stage, python:3.12-slim, uv, uvicorn)
- `.dockerignore` (3.2: excluye .venv, .git, .env, qdrant_db, __pycache__)
- `HEALTHCHECK` en el Dockerfile (3.2)
- `.github/workflows/ci.yml` (4.1: mypy + pytest; 4.2: docker build)
- `README.md` en inglés (5.1: qué hace, run local, URL pública, tablas de evals y RAG)

Verifica SIEMPRE: ninguna API key dentro de la imagen ni en el repo.
Ver soluciones completas en ejercicios_deploy_fastapi_docker_ci.MD (Nivel 3-5).
