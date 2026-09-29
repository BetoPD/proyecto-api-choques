"""API mínima que expone el modelo de severidad de choques (ver TDD.md).

Ejecutar:  uvicorn app.main:app --reload --port 8000   ->  http://localhost:8000/docs
"""
import sys
from pathlib import Path

import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# El pickle referencia el módulo "severidad_features", así que debe ser importable por ese nombre.
APP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(APP_DIR))
import severidad_features as sf  # noqa: E402

MODELO = joblib.load(APP_DIR.parent / "modelo" / "modelo_champion.pkl")

app = FastAPI(title="API de severidad de choques CDMX", version="1.0.0")


class Reporte(BaseModel):
    """Primer aviso de un siniestro. Los vehículos no enviados cuentan como 0."""
    MES: int
    DIASEMANA: str
    ID_HORA: int
    ALCALDIA: str
    TIPACCID: str
    URBANA: str
    AUTOMOVIL: int = 0
    CAMPASAJ: int = 0
    MICROBUS: int = 0
    PASCAMION: int = 0
    OMNIBUS: int = 0
    TRANVIA: int = 0
    CAMIONETA: int = 0
    CAMION: int = 0
    TRACTOR: int = 0
    MOTOCICLET: int = 0
    BICICLETA: int = 0
    OTROVEHIC: int = 0
    FERROCARRI: int = 0

    model_config = {"json_schema_extra": {"examples": [{
        "MES": 3, "DIASEMANA": "Sábado", "ID_HORA": 2, "ALCALDIA": "Iztapalapa",
        "TIPACCID": "Colisión con peatón (atropellamiento)", "URBANA": "No intersección", "AUTOMOVIL": 1,
    }]}}


class Prediccion(BaseModel):
    P_S0: float
    P_S1: float
    P_S2: float
    score: float
    prioridad: str
    clase_probable: str


@app.get("/health")
def health():
    return {"status": "ok", "modelo": MODELO.metadatos.get("modelo"),
            "ventana_entrenamiento": MODELO.metadatos.get("ventana_entrenamiento"),
            "entrenado": MODELO.metadatos.get("entrenado")}


@app.post("/predict", response_model=Prediccion | list[Prediccion])
def predict(entrada: Reporte | list[Reporte]):
    registros = [r.model_dump() for r in entrada] if isinstance(entrada, list) else entrada.model_dump()
    datos, errores = sf.validar_reporte(registros)
    if errores:
        raise HTTPException(status_code=422, detail={"errores": errores})

    salida = MODELO.predecir(datos).round(4)
    predicciones = [Prediccion(P_S0=r.P_S0, P_S1=r.P_S1, P_S2=r.P_S2, score=r.SCORE,
                               prioridad=r.PRIORIDAD, clase_probable=r.CLASE_PROBABLE)
                    for r in salida.itertuples()]
    return predicciones if isinstance(entrada, list) else predicciones[0]
