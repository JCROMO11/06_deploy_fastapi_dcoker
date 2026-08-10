"""
Nivel 1 — FastAPI básico
Setup: uv add fastapi uvicorn[standard] anthropic pydantic
Levanta con: uvicorn 01_fastapi_basico:app --reload  ->  abre /docs
"""
from fastapi import FastAPI
from pydantic import BaseModel

# 1.1 — app = FastAPI(); GET /health -> {"status":"ok"}
app = FastAPI()

@app.get('/health')
def get_status():
    return {'status':'ok'}

# 1.2 — PreguntaIn{pregunta} y RespuestaOut{respuesta, fuentes}; POST /preguntar (respuesta fija)
class PreguntaIn(BaseModel):
    pregunta: str

class RespuestaOut(BaseModel):
    respuesta: str
    fuentes: list[str]
    
@app.post('/preguntar')
def ask(question: PreguntaIn) -> RespuestaOut:
    return RespuestaOut(respuesta='I am ur answer', fuentes=['google', 'openai', 'anthropic'])
