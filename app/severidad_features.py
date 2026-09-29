"""Funciones compartidas entre el notebook y la API de severidad (Equipo 3, UP).

Contiene: etiqueta S0/S1/S2, variables disponibles en el primer aviso, preprocesador,
calibración multiclase, reglas de prioridad, validación de reportes y el objeto
ModeloSeveridad que se serializa en modelo_champion.pkl.
"""
import unicodedata

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler

CLASES = ["S0", "S1", "S2"]
PRIORIDADES = ["Baja", "Media", "Alta"]

ALCALDIAS = {
    "09002": "Azcapotzalco", "09003": "Coyoacán", "09004": "Cuajimalpa de Morelos",
    "09005": "Gustavo A. Madero", "09006": "Iztacalco", "09007": "Iztapalapa",
    "09008": "La Magdalena Contreras", "09009": "Milpa Alta", "09010": "Álvaro Obregón",
    "09011": "Tláhuac", "09012": "Tlalpan", "09013": "Xochimilco", "09014": "Benito Juárez",
    "09015": "Cuauhtémoc", "09016": "Miguel Hidalgo", "09017": "Venustiano Carranza",
}

VICTIMAS_HERIDAS = ["CONDHERIDO", "PASAHERIDO", "PEATHERIDO", "CICLHERIDO", "OTROHERIDO", "NEHERIDO"]
VICTIMAS_MUERTAS = ["CONDMUERTO", "PASAMUERTO", "PEATMUERTO", "CICLMUERTO", "OTROMUERTO", "NEMUERTO"]

GRUPOS_VEHICULO = {
    "N_AUTOS": ["AUTOMOVIL", "CAMPASAJ"],
    "N_TRANSP_PUBLICO": ["MICROBUS", "PASCAMION", "OMNIBUS", "TRANVIA"],
    "N_CARGA": ["CAMIONETA", "CAMION", "TRACTOR"],
    "N_MOTO": ["MOTOCICLET"],
    "N_BICI": ["BICICLETA"],
    "N_OTRO": ["OTROVEHIC", "FERROCARRI"],
}
VEHICULOS = [c for grupo in GRUPOS_VEHICULO.values() for c in grupo]

DIAS = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"]
TIPOS_ACCIDENTE = [
    "Colisión con vehículo automotor", "Colisión con peatón (atropellamiento)", "Colisión con animal",
    "Colisión con objeto fijo", "Volcadura", "Caída de pasajero", "Salida del camino", "Incendio",
    "Colisión con ferrocarril", "Colisión con motocicleta", "Colisión con ciclista", "Otro",
]
UBICACIONES = {"Accidente en intersección": "Intersección", "Accidente en no intersección": "No intersección"}
FRANJAS = ["Madrugada (0-5)", "Mañana (6-11)", "Tarde (12-17)", "Noche (18-23)"]

FEATURES_CAT = ["MES", "DIASEMANA", "FRANJA", "ALCALDIA", "TIPACCID", "UBICACION"]
FEATURES_NUM = ["HORA_SIN", "HORA_COS", "FIN_SEMANA", "NVEH", *GRUPOS_VEHICULO,
                "HAY_MOTO", "HAY_BICI", "HAY_TRANSP_PUBLICO", "HAY_CARGA_PESADA", "UN_VEHICULO"]


# ----------------------------------------------------------------------------- etiqueta
def construir_objetivo(df):
    """S2 si hay al menos un fallecido en el lugar; S1 si hay heridos y ningún fallecido; S0 en otro caso."""
    heridos = df[VICTIMAS_HERIDAS].apply(pd.to_numeric, errors="coerce").clip(lower=0).sum(axis=1)
    muertos = df[VICTIMAS_MUERTAS].apply(pd.to_numeric, errors="coerce").clip(lower=0).sum(axis=1)
    return pd.Series(np.select([muertos >= 1, heridos >= 1], ["S2", "S1"], "S0"), index=df.index, name="SEVERIDAD")


# ----------------------------------------------------------------------------- variables
def _sin_acentos(texto):
    texto = unicodedata.normalize("NFKD", str(texto))
    return "".join(c for c in texto if not unicodedata.combining(c)).strip().lower()

_ALCALDIA_POR_TEXTO = {_sin_acentos(v): v for v in ALCALDIAS.values()}
_ALCALDIA_POR_TEXTO.update({k: v for k, v in ALCALDIAS.items()})

def normalizar_alcaldia(valor):
    """Acepta CVEGEO ('09007'), nombre con o sin acentos ('Iztapalapa', 'alvaro obregon')."""
    if valor is None:
        return None
    v = str(valor).strip()
    if v in ALCALDIAS:
        return ALCALDIAS[v]
    return _ALCALDIA_POR_TEXTO.get(_sin_acentos(v))

def franja_horaria(hora):
    etiquetas = pd.cut(hora, [-1, 5, 11, 17, 23], labels=FRANJAS)
    return etiquetas.astype(object).where(etiquetas.notna(), "No especificada").astype(str)

def construir_features(df):
    """Variables observables en el primer aviso del siniestro (Etapa 1).

    Funciona con registros ATUS (usa CVEGEO) o con reportes de la API (usa ALCALDIA).
    """
    X = pd.DataFrame(index=df.index)
    hora = pd.to_numeric(df["ID_HORA"], errors="coerce")
    hora = hora.where(hora.between(0, 23))
    X["MES"] = pd.to_numeric(df["MES"], errors="coerce").fillna(0).astype(int).astype(str)
    dia = df["DIASEMANA"].astype(str)
    X["DIASEMANA"] = dia.where(dia.isin(DIAS), "No especificado")
    X["FIN_SEMANA"] = dia.isin(["Sábado", "Domingo"]).astype(int)
    X["FRANJA"] = franja_horaria(hora)
    X["HORA_SIN"] = np.sin(2 * np.pi * hora.fillna(12) / 24)
    X["HORA_COS"] = np.cos(2 * np.pi * hora.fillna(12) / 24)
    if "ALCALDIA" in df.columns:
        alcaldia = df["ALCALDIA"].map(normalizar_alcaldia)
    else:
        alcaldia = df["CVEGEO"].astype(str).str.zfill(5).map(ALCALDIAS)
    X["ALCALDIA"] = alcaldia.fillna("Desconocida").astype(str)
    X["TIPACCID"] = df["TIPACCID"].astype(str)
    X["UBICACION"] = df["URBANA"].map(UBICACIONES).fillna("Suburbana u otra").astype(str)

    veh = pd.DataFrame({c: (pd.to_numeric(df[c], errors="coerce").fillna(0).clip(lower=0)
                            if c in df.columns else 0) for c in VEHICULOS}, index=df.index)
    for grupo, cols in GRUPOS_VEHICULO.items():
        X[grupo] = veh[cols].sum(axis=1)
    X["NVEH"] = veh.sum(axis=1)
    X["HAY_MOTO"] = (X["N_MOTO"] > 0).astype(int)
    X["HAY_BICI"] = (X["N_BICI"] > 0).astype(int)
    X["HAY_TRANSP_PUBLICO"] = (X["N_TRANSP_PUBLICO"] > 0).astype(int)
    X["HAY_CARGA_PESADA"] = ((veh["CAMION"] + veh["TRACTOR"]) > 0).astype(int)
    X["UN_VEHICULO"] = (X["NVEH"] == 1).astype(int)
    return X[FEATURES_CAT + FEATURES_NUM]

def crear_preprocesador():
    """One-hot para categóricas (categorías con < 30 casos se agrupan) y estandarización de numéricas."""
    return ColumnTransformer(
        [("cat", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=30, sparse_output=False), FEATURES_CAT),
         ("num", StandardScaler(), FEATURES_NUM)],
        verbose_feature_names_out=False,
    )


# ----------------------------------------------------------------------------- modelo
class ClasificadorSeveridad:
    """Preprocesador ajustado + estimador. predict_proba recibe registros crudos (ATUS o API)."""
    def __init__(self, preprocesador, modelo, nombre=""):
        self.preprocesador, self.modelo, self.nombre = preprocesador, modelo, nombre

    def transformar(self, df):
        return self.preprocesador.transform(construir_features(df))

    def predict_proba(self, df):
        return self.modelo.predict_proba(self.transformar(df))


class CalibradorMultinomial:
    """Escalamiento de Platt multiclase: regresión logística sobre log-probabilidades.

    Corrige el sesgo que introducen los pesos por clase (que inflan S1 y S2) para que
    P(S0), P(S1) y P(S2) puedan leerse como probabilidades.
    """
    def __init__(self, C=10.0):
        self.C = C

    @staticmethod
    def _z(P):
        return np.log(np.clip(np.asarray(P, dtype=float), 1e-6, 1.0))

    def fit(self, P, y):
        self.lr_ = LogisticRegression(C=self.C, max_iter=5000).fit(self._z(P), y)
        return self

    def transform(self, P):
        return self.lr_.predict_proba(self._z(P))


def score_severidad(P):
    """SSE = P(S1) + 2 P(S2); score 0-100 = 50 * SSE (sección 6.3)."""
    P = np.asarray(P)
    return 50 * (P[:, 1] + 2 * P[:, 2])

def asignar_prioridad(P, umbrales):
    P = np.asarray(P)
    score = score_severidad(P)
    alta = (P[:, 2] >= umbrales["p_s2_alta"]) | (score >= umbrales["score_alta"])
    media = ~alta & (score >= umbrales["score_media"])
    return np.where(alta, "Alta", np.where(media, "Media", "Baja"))


class ModeloSeveridad:
    """Objeto champion que se despliega: clasificador + calibrador + umbrales de prioridad."""
    def __init__(self, clasificador, calibrador, umbrales, metadatos=None):
        self.clasificador, self.calibrador, self.umbrales = clasificador, calibrador, umbrales
        self.metadatos = metadatos or {}

    def probabilidades(self, df, calibradas=True):
        P = self.clasificador.predict_proba(df)
        return self.calibrador.transform(P) if calibradas and self.calibrador is not None else P

    def predecir(self, df):
        P_modelo = self.clasificador.predict_proba(df)
        P = self.calibrador.transform(P_modelo) if self.calibrador is not None else P_modelo
        salida = pd.DataFrame(P, columns=[f"P_{c}" for c in CLASES], index=df.index)
        salida["SSE"] = P[:, 1] + 2 * P[:, 2]
        salida["SCORE"] = score_severidad(P)
        salida["PRIORIDAD"] = asignar_prioridad(P, self.umbrales)
        salida["CLASE_PROBABLE"] = np.array(CLASES)[P_modelo.argmax(axis=1)]
        return salida


# ----------------------------------------------------------------------------- validación API
CAMPOS_REQUERIDOS = ["MES", "DIASEMANA", "ID_HORA", "ALCALDIA", "TIPACCID", "URBANA"]
_UBICACION_TEXTO = {**{k: k for k in UBICACIONES}, **{v: k for k, v in UBICACIONES.items()}}

def validar_reporte(registros):
    """Valida uno o varios reportes de primer aviso. Devuelve (DataFrame o None, lista de errores)."""
    if isinstance(registros, dict):
        registros = [registros]
    filas, errores = [], []
    for i, r in enumerate(registros):
        e = []
        faltan = [c for c in CAMPOS_REQUERIDOS if c not in r]
        if faltan:
            errores.append({"registro": i, "errores": [f"Faltan campos: {', '.join(faltan)}"]})
            continue
        fila = {}
        try:
            fila["MES"] = int(r["MES"]); assert 1 <= fila["MES"] <= 12
        except Exception:
            e.append("MES debe ser un entero entre 1 y 12")
        try:
            fila["ID_HORA"] = int(r["ID_HORA"]); assert 0 <= fila["ID_HORA"] <= 23
        except Exception:
            e.append("ID_HORA debe ser un entero entre 0 y 23")
        if r["DIASEMANA"] not in DIAS:
            e.append(f"DIASEMANA debe ser uno de {DIAS}")
        fila["DIASEMANA"] = r["DIASEMANA"]
        alcaldia = normalizar_alcaldia(r["ALCALDIA"])
        if alcaldia is None:
            e.append("ALCALDIA no reconocida (usa el nombre de la alcaldía o su CVEGEO)")
        fila["ALCALDIA"] = alcaldia
        if r["TIPACCID"] not in TIPOS_ACCIDENTE:
            e.append(f"TIPACCID debe ser uno de {TIPOS_ACCIDENTE}")
        fila["TIPACCID"] = r["TIPACCID"]
        if r["URBANA"] not in _UBICACION_TEXTO:
            e.append("URBANA debe ser 'Intersección' o 'No intersección'")
        fila["URBANA"] = _UBICACION_TEXTO.get(r["URBANA"])
        total = 0
        for c in VEHICULOS:
            try:
                fila[c] = int(r.get(c, 0)); assert 0 <= fila[c] <= 9
                total += fila[c]
            except Exception:
                e.append(f"{c} debe ser un entero entre 0 y 9")
        if total == 0:
            e.append("El reporte debe incluir al menos un vehículo")
        if e:
            errores.append({"registro": i, "errores": e})
        else:
            filas.append(fila)
    return (pd.DataFrame(filas) if not errores else None), errores
