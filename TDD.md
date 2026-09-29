# TDD — Crash Severity API (proyecto-api-choques)

Technical Design Document for a minimal Python API that exposes the crash-severity
model trained in `proyecto-final-data-science`.

## 1. Purpose and scope

The goal is to show, as simply as possible, how a trained ML model is **exposed over HTTP
in Python** so other systems can consume it.

- **In scope:** load the serialized model, validate a request, return predictions as JSON.
- **Out of scope (demo only):** authentication, persistence or logging to a database,
  scaling, containerization, monitoring.

## 2. The model

| Item | Value |
|---|---|
| Artifact | `modelo/modelo_champion.pkl` (joblib), a `ModeloSeveridad` object |
| Code it depends on | `app/severidad_features.py` (it must be importable to unpickle) |
| Estimator | XGBoost + multinomial calibration + priority thresholds |
| Training window | 2022–2023 (post-pandemic), validated on 2024, tested on 2025 |
| Data | INEGI ATUS, CDMX road accidents 2019–2025 |
| Library versions | scikit-learn 1.9.1, xgboost 3.4.1, pandas 3.0.6, Python 3.12 |

**Task:** given the *first notice* of a crash (what's known when it gets reported), estimate
how severe it is:

- **S0**: no victims
- **S1**: injured people, no deaths
- **S2**: at least one death at the scene

**Outputs per report:**

- `P_S0`, `P_S1`, `P_S2`: calibrated probabilities (they add up to 1).
- `score`: 0–100 severity score = `50 · (P(S1) + 2·P(S2))`.
- `prioridad`: `Alta` if `P(S2) ≥ 0.14` **or** `score ≥ 50.24`; `Media` if `score ≥ 10.75`; otherwise `Baja`.
- `clase_probable`: the class with the highest raw (uncalibrated) model probability.

**Test-set metrics (2025):** Macro F1 0.568 · Recall S1 0.550 · Recall S2 0.486 ·
PR-AUC S2 0.192 · ROC-AUC (any victim) 0.900.

## 3. Architecture

```
client ──HTTP/JSON──▶ FastAPI (app/main.py)
                        │ 1. Pydantic: types and required fields
                        │ 2. sf.validar_reporte(): domain rules (ranges, catalogs)
                        │ 3. MODELO.predecir(df): features → XGBoost → calibration → priority
                        ▼
                      JSON response
```

- **Framework:** FastAPI served by uvicorn. It's small, and the interactive docs at `/docs` come for free.
- The model is **loaded once at startup** (module import), not on every request.
- We reuse the project's code (`validar_reporte`, `construir_features`, `predecir`) so the
  API applies **exactly** the same preprocessing as training.

### Project layout

```
proyecto-api-choques/
├── TDD.md                     # this document
├── README.md
├── requirements.txt
├── app/
│   ├── main.py                # the API
│   └── severidad_features.py  # copied from the training project
├── modelo/
│   └── modelo_champion.pkl    # copied from the training project
└── tests/
    └── test_api.py
```

## 4. Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Liveness check plus model metadata |
| `POST` | `/predict` | Predicts severity for one report (object) or several (array) |
| `GET` | `/docs` | Swagger UI, generated automatically |

### 4.1 `GET /health`

Response `200`:

```json
{
  "status": "ok",
  "modelo": "XGBoost",
  "ventana_entrenamiento": "B: 2022-2023 (pospandemia)",
  "entrenado": "2026-09-28 19:15"
}
```

### 4.2 `POST /predict`

**Body:** one JSON object (a report) **or** a JSON array of reports.
Field names follow the INEGI ATUS dataset.

| Field | Type | Required | Allowed values |
|---|---|---|---|
| `MES` | int | yes | 1–12 |
| `DIASEMANA` | string | yes | `Lunes`, `Martes`, `Miércoles`, `Jueves`, `Viernes`, `Sábado`, `Domingo` |
| `ID_HORA` | int | yes | 0–23 |
| `ALCALDIA` | string | yes | Borough name (accents optional, e.g. `Iztapalapa`, `alvaro obregon`) or CVEGEO (`09007`) |
| `TIPACCID` | string | yes | See the list below |
| `URBANA` | string | yes | `Intersección` or `No intersección` |
| `AUTOMOVIL`, `CAMPASAJ`, `MICROBUS`, `PASCAMION`, `OMNIBUS`, `TRANVIA`, `CAMIONETA`, `CAMION`, `TRACTOR`, `MOTOCICLET`, `BICICLETA`, `OTROVEHIC`, `FERROCARRI` | int | no (default 0) | 0–9 each; **at least one vehicle in total** |

`TIPACCID` values: `Colisión con vehículo automotor`, `Colisión con peatón (atropellamiento)`,
`Colisión con animal`, `Colisión con objeto fijo`, `Volcadura`, `Caída de pasajero`,
`Salida del camino`, `Incendio`, `Colisión con ferrocarril`, `Colisión con motocicleta`,
`Colisión con ciclista`, `Otro`.

**Response `200`:** one object for an object input, or an array (same order) for an array input.
This is the real output for the single-report example in §5 (a pedestrian hit at 2 a.m. in Iztapalapa):

```json
{
  "P_S0": 0.0096,
  "P_S1": 0.7556,
  "P_S2": 0.2348,
  "score": 61.2609,
  "prioridad": "Alta",
  "clase_probable": "S2"
}
```

### 4.3 Errors

| Status | When | Body |
|---|---|---|
| `422` | Wrong types or missing fields (Pydantic) | FastAPI's standard `{"detail": [...]}` |
| `422` | Domain rules fail (out-of-range month, unknown borough, no vehicles…) | `{"detail": {"errores": [{"registro": 0, "errores": ["..."]}]}}` |

Example:

```json
{"detail": {"errores": [{"registro": 0, "errores": ["MES debe ser un entero entre 1 y 12"]}]}}
```

## 5. How to make a request

Start the server first (see §6). The base URL is `http://localhost:8000`.

**Health check**

```bash
curl http://localhost:8000/health
```

**Single report (curl)**

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
        "MES": 3,
        "DIASEMANA": "Sábado",
        "ID_HORA": 2,
        "ALCALDIA": "Iztapalapa",
        "TIPACCID": "Colisión con peatón (atropellamiento)",
        "URBANA": "No intersección",
        "AUTOMOVIL": 1
      }'
```

**Several reports in one call (curl)**

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '[
        {"MES": 3, "DIASEMANA": "Sábado", "ID_HORA": 2, "ALCALDIA": "Iztapalapa",
         "TIPACCID": "Colisión con peatón (atropellamiento)", "URBANA": "No intersección", "AUTOMOVIL": 1},
        {"MES": 7, "DIASEMANA": "Martes", "ID_HORA": 14, "ALCALDIA": "09014",
         "TIPACCID": "Colisión con vehículo automotor", "URBANA": "Intersección", "AUTOMOVIL": 2}
      ]'
```

**Python (`requests`)**

```python
import requests

reporte = {
    "MES": 3, "DIASEMANA": "Sábado", "ID_HORA": 2, "ALCALDIA": "Iztapalapa",
    "TIPACCID": "Colisión con peatón (atropellamiento)", "URBANA": "No intersección",
    "AUTOMOVIL": 1,
}
r = requests.post("http://localhost:8000/predict", json=reporte, timeout=10)
r.raise_for_status()
print(r.json())   # {'P_S0': ..., 'P_S1': ..., 'P_S2': ..., 'score': ..., 'prioridad': ..., 'clase_probable': ...}
```

**Browser:** open <http://localhost:8000/docs>, expand `POST /predict`, click
*Try it out*, and send the example body.

## 6. How to run

```bash
cd proyecto-api-choques
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

## 7. Testing

`tests/test_api.py` uses FastAPI's `TestClient` (no server needed):

- `/health` returns 200 and names the model.
- A valid single report returns 200, the probabilities add up to about 1, and `prioridad` is one of Baja/Media/Alta.
- An array of 2 reports returns an array of 2 predictions.
- An out-of-range `MES` returns 422.
- A report with no vehicles returns 422.

```bash
pytest -q
```

## 8. Limitations and notes

- **Version pinning matters:** the pickle has to be loaded with the same scikit-learn/xgboost
  versions it was trained with (see `requirements.txt`).
- **The pickle depends on the code:** `severidad_features.py` must stay importable as a top-level
  module named `severidad_features`. If the model gets retrained, copy both files again.
- Loading a pickle runs code, so only load model files you trust.
- There's no auth, rate limiting or request logging. This is a teaching demo, not a production service.
