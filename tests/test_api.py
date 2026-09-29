from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

REPORTE = {
    "MES": 3, "DIASEMANA": "Sábado", "ID_HORA": 2, "ALCALDIA": "Iztapalapa",
    "TIPACCID": "Colisión con peatón (atropellamiento)", "URBANA": "No intersección", "AUTOMOVIL": 1,
}


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["modelo"] == "XGBoost"


def test_predict_un_reporte():
    r = client.post("/predict", json=REPORTE)
    assert r.status_code == 200
    p = r.json()
    assert abs(p["P_S0"] + p["P_S1"] + p["P_S2"] - 1) < 1e-3
    assert p["prioridad"] in {"Baja", "Media", "Alta"}
    assert p["clase_probable"] in {"S0", "S1", "S2"}


def test_predict_varios_reportes():
    otro = {**REPORTE, "ALCALDIA": "09014", "TIPACCID": "Colisión con vehículo automotor", "AUTOMOVIL": 2}
    r = client.post("/predict", json=[REPORTE, otro])
    assert r.status_code == 200
    assert len(r.json()) == 2


def test_mes_invalido():
    r = client.post("/predict", json={**REPORTE, "MES": 13})
    assert r.status_code == 422
    assert "errores" in r.json()["detail"]


def test_sin_vehiculos():
    r = client.post("/predict", json={**REPORTE, "AUTOMOVIL": 0})
    assert r.status_code == 422
