# proyecto-api-choques

Minimal FastAPI service that exposes the CDMX crash-severity model (S0/S1/S2 + priority).
The full design and **how to make requests** are in [TDD.md](TDD.md).

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000   # docs at http://localhost:8000/docs
pytest -q
```
