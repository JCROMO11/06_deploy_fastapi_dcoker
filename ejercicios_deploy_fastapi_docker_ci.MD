# 🚢 Ejercicios: Deploy — FastAPI + Docker + CI

> **Cómo usar este archivo**
>
> 1. Un archivo/carpeta por nivel en `06_deploy_fastapi_docker_ci/ejercicios/`.
> 2. Resuelve **sin mirar la solución** (al final).
> 3. **Diferenciador #2 del roadmap:** *"Notebooks muestran data science; paquetes deployables muestran
>    ingeniería."* El entregable de la Semana 7 es el Proyecto 1 completo: API REST dockerizada, con CI y URL pública.
>
> **Setup:**
>
> ```bash
> uv add fastapi uvicorn[standard] anthropic pydantic
> # Docker instalado. Cuenta gratuita en un PaaS (Render/Railway/Fly.io) para la URL pública.
> ```

---

## 🟢 NIVEL 1 — FastAPI básico

### Ejercicio 1.1 — Tu primer endpoint

**Objetivo:** levantar una API y ver la doc automática.

**Enunciado:** Crea `main.py` con una app FastAPI y un endpoint `GET /health` que devuelva `{"status": "ok"}`. Levántala con `uvicorn main:app --reload` y abre `/docs` (Swagger automático).

**Pista:** `app = FastAPI()`; `@app.get("/health")`. La doc en `/docs` sale gratis — menciónalo en entrevistas.

---

### Ejercicio 1.2 — Request/response tipados con Pydantic

**Objetivo:** validación automática de entrada y salida.

**Enunciado:** Crea `POST /preguntar` que reciba `PreguntaIn(BaseModel){pregunta: str}` y devuelva `RespuestaOut(BaseModel){respuesta: str, fuentes: list[str]}`. Por ahora devuelve una respuesta fija. Prueba en `/docs` que un body inválido da 422 automático.

**Pista:** FastAPI valida el body con tu modelo Pydantic sin código extra. Aquí se nota por qué estudiaste Pydantic.

---

## 🟡 NIVEL 2 — Integrar tu RAG + config + errores

### Ejercicio 2.1 — Conectar el RAG real

**Objetivo:** exponer tu RAG (Proyecto 1) por HTTP.

**Enunciado:** Importa tu `responder()` del módulo RAG y conéctalo a `POST /preguntar`. La API ahora responde de verdad usando retrieval + LLM.

**Pista:** carga el índice una sola vez al arrancar (no por request). Usa el evento de startup o una dependencia cacheada.

---

### Ejercicio 2.2 — Settings y manejo de errores HTTP

**Objetivo:** config tipada (módulo 05 de Fase 1) + errores limpios.

**Enunciado:** Usa `pydantic-settings` para leer `ANTHROPIC_API_KEY` y parámetros (modelo, top_k). Envuelve la llamada al RAG en try/except y devuelve `HTTPException(503)` si el LLM falla, en vez de un 500 crudo.

**Pista:** `from fastapi import HTTPException`. La key NUNCA hardcodeada — viene del `.env` (que está en `.gitignore`).

---

### Ejercicio 2.3 — Streaming de la respuesta

**Objetivo:** UX de "se va escribiendo" sobre HTTP.

**Enunciado:** Crea `POST /preguntar/stream` que use `StreamingResponse` para enviar la respuesta del LLM por chunks (reutiliza el streaming del módulo 01).

**Pista:** `from fastapi.responses import StreamingResponse`; un generador que haga `yield` de cada fragmento.

---

## 🟠 NIVEL 3 — Dockerizar

### Ejercicio 3.1 — Dockerfile multi-stage

**Objetivo:** imagen pequeña y reproducible.

**Enunciado:** Escribe un `Dockerfile` (base `python:3.12-slim`) que instale deps con `uv`, copie el código y arranque `uvicorn`. Constrúyela (`docker build`) y córrela (`docker run -p 8000:8000`), pasando la key con `-e`.

**Pista:** copia primero los archivos de deps (para cachear la capa de instalación) y luego el código. Expón el puerto 8000.

---

### Ejercicio 3.2 — .dockerignore y healthcheck

**Objetivo:** imagen limpia y observable.

**Enunciado:** Añade un `.dockerignore` (excluye `.venv`, `.git`, `__pycache__`, `.env`, `qdrant_db`) y un `HEALTHCHECK` en el Dockerfile que pegue a `/health`. Verifica que `.env` NO quede dentro de la imagen.

**Pista:** `docker history` o `docker run ... cat .env` para confirmar que el secreto no se filtró. Esto es del checklist del roadmap.

---

## 🔴 NIVEL 4 — CI con GitHub Actions

### Ejercicio 4.1 — CI: lint + tests

**Objetivo:** que cada push verifique el código.

**Enunciado:** Crea `.github/workflows/ci.yml` que en cada push: instale deps, corra `mypy` (del módulo type hints) y `pytest`. Que falle el build si algo no pasa.

**Pista:** usa `astral-sh/setup-uv`. Los tests que pegan a la API real necesitan la key como **secret**; considera mockearlos o marcarlos.

---

### Ejercicio 4.2 — CI: build de la imagen Docker

**Objetivo:** verificar que la imagen siempre construye.

**Enunciado:** Añade un job que haga `docker build` (y opcionalmente push a GHCR). Así nunca se rompe el deploy por una imagen que no compila.

**Pista:** `docker/build-push-action`. Para GHCR usa `GITHUB_TOKEN`.

---

## 🟣 NIVEL 5 — Integrador: deploy con URL pública

### Ejercicio 5.1 — RETO: Proyecto 1 deployado y vivo

**Objetivo:** el **entregable de la Semana 7** — URL pública funcional.

**Enunciado:**

1. Sube el repo a GitHub (verifica: **cero keys commiteadas**).
2. Conéctalo a un PaaS gratis (Render/Railway/Fly.io), configura la key como variable de entorno en el panel.
3. Deploya. Verifica que `/docs` y `/preguntar` responden en la URL pública.
4. Escribe el `README.md` (en inglés) con: qué hace, cómo correrlo local, la URL, y la tabla de evals (módulo 05) y de RAG avanzado (módulo 04).
   Documenta una decisión de arquitectura (p.ej. "RAG vs fine-tuning: por qué RAG aquí").

**Pista:** el README es tan importante como el código — es lo primero que ve un reclutador. Incluye un GIF/captura.

**Por qué importa:** con esto cierras el **Proyecto 1** y marcas 4 casillas del checklist de empleabilidad de
una vez: repo de producción + CI + URL pública + sin keys commiteadas. Es el primero de los 3 proyectos que
te llevan al 70-90%.

---

---

# ✅ SOLUCIONES

> Intenta TODO antes de mirar.

### Solución 1.1

```python
# main.py
from fastapi import FastAPI

app = FastAPI(title="RAG API")

@app.get("/health")
def health():
    return {"status": "ok"}
# uvicorn main:app --reload  ->  abre http://localhost:8000/docs
```

### Solución 1.2

```python
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class PreguntaIn(BaseModel):
    pregunta: str

class RespuestaOut(BaseModel):
    respuesta: str
    fuentes: list[str]

@app.post("/preguntar", response_model=RespuestaOut)
def preguntar(body: PreguntaIn):
    return RespuestaOut(respuesta=f"Echo: {body.pregunta}", fuentes=[])
```

### Solución 2.1

```python
from contextlib import asynccontextmanager
from fastapi import FastAPI
from rag import indexar, responder   # tu módulo RAG

@asynccontextmanager
async def lifespan(app: FastAPI):
    indexar("corpus.txt")   # carga el índice una vez al arrancar
    yield

app = FastAPI(lifespan=lifespan)

@app.post("/preguntar", response_model=RespuestaOut)
def preguntar(body: PreguntaIn):
    r = responder(body.pregunta)
    return RespuestaOut(respuesta=r.respuesta, fuentes=[str(f) for f in r.fuentes])
```

### Solución 2.2

```python
from fastapi import FastAPI, HTTPException
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env")
    anthropic_api_key: str
    modelo: str = "claude-haiku-4-5-20251001"
    top_k: int = 3

settings = Settings()

@app.post("/preguntar", response_model=RespuestaOut)
def preguntar(body: PreguntaIn):
    try:
        r = responder(body.pregunta)
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"El servicio LLM falló: {e}")
    return RespuestaOut(respuesta=r.respuesta, fuentes=[str(f) for f in r.fuentes])
```

### Solución 2.3

```python
from fastapi.responses import StreamingResponse
import anthropic

client = anthropic.Anthropic()

@app.post("/preguntar/stream")
def preguntar_stream(body: PreguntaIn):
    def gen():
        with client.messages.stream(model="claude-haiku-4-5-20251001", max_tokens=500,
                messages=[{"role": "user", "content": body.pregunta}]) as stream:
            for text in stream.text_stream:
                yield text
    return StreamingResponse(gen(), media_type="text/plain")
```

### Solución 3.1 — Dockerfile

```dockerfile
FROM python:3.12-slim
WORKDIR /app
RUN pip install --no-cache-dir uv
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev
COPY . .
EXPOSE 8000
CMD ["uv", "run", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

```bash
docker build -t rag-api .
docker run -p 8000:8000 -e ANTHROPIC_API_KEY="$ANTHROPIC_API_KEY" rag-api
```

### Solución 3.2 — .dockerignore + healthcheck

```
# .dockerignore
.venv
.git
__pycache__
.env
qdrant_db
*.pyc
```

```dockerfile
HEALTHCHECK --interval=30s --timeout=3s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
```

### Solución 4.1 — .github/workflows/ci.yml

```yaml
name: ci
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v3
      - run: uv sync
      - run: uv run mypy .
      - name: tests
        env:
          ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
        run: uv run pytest -v
```

### Solución 4.2 — job de build

```yaml
  docker:
    runs-on: ubuntu-latest
    needs: test
    steps:
      - uses: actions/checkout@v4
      - uses: docker/build-push-action@v6
        with:
          context: .
          push: false   # ponlo en true + login a GHCR cuando quieras publicar
          tags: rag-api:latest
```

### Solución 5.1 (RETO)

```text
Checklist de cierre del Proyecto 1:
[ ] git: ningún .env ni key en el historial (revisa con: git log -p | grep -i sk-ant)
[ ] PaaS conectado al repo, ANTHROPIC_API_KEY configurada en el panel (no en el código)
[ ] /health, /docs y /preguntar responden en la URL pública
[ ] README en inglés: qué hace, run local, URL, tabla de evals, tabla RAG antes/después
[ ] README responde "RAG vs fine-tuning: por qué elegí esto"
[ ] GIF o captura de la API funcionando
```

---

## 🎯 Cómo medir tu progreso

- **Niveles 1-2:** tienes una API tipada que sirve tu RAG con manejo de errores. Ya es ingeniería, no notebook.
- **Nivel 3:** dockerizaste sin filtrar secretos. Reproducible en cualquier máquina.
- **Nivel 4:** CI que verifica cada push. Calidad automatizada.
- **Nivel 5:** **Proyecto 1 vivo con URL pública.** Primer proyecto de producción del portfolio. 🚢

Frase para entrevistas: *"No es un notebook — es un paquete deployable, con CI y una URL que puedes abrir ahora."*
