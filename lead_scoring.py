"""
Modelo de lead scoring con calificación MQL / SQL.

Flujo:
1. Genera una base sintética de leads B2B (comportamiento + perfil).
2. Entrena una regresión logística que estima la probabilidad de que el lead convierta.
3. Convierte esa probabilidad en un puntaje de 0 a 100.
4. Aplica reglas de negocio para clasificar cada lead como SQL, MQL o Nutrición.
5. Guarda el modelo, la base calificada y una gráfica de validación.

Los datos son sintéticos: sirven para demostrar el método, no describen clientes reales.
"""

from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SALIDA = Path(__file__).parent / "resultados"
SALIDA.mkdir(exist_ok=True)

NUMERICAS = [
    "visitas_web",
    "paginas_vistas",
    "descargas_contenido",
    "emails_abiertos",
    "clics_email",
    "dias_ultima_interaccion",
]
BINARIAS = ["lleno_formulario", "cargo_decisor"]
CATEGORICAS = ["tamano_empresa", "sector", "fuente"]

# Umbrales de negocio: se ajustan con el equipo comercial según capacidad de ventas.
UMBRAL_SQL = 70
UMBRAL_MQL = 40


def generar_leads(n: int = 3000, semilla: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(semilla)
    df = pd.DataFrame(
        {
            "visitas_web": rng.poisson(4, n),
            "paginas_vistas": rng.poisson(9, n),
            "descargas_contenido": rng.poisson(0.8, n),
            "emails_abiertos": rng.poisson(3, n),
            "clics_email": rng.poisson(1, n),
            "dias_ultima_interaccion": rng.integers(0, 90, n),
            "lleno_formulario": rng.binomial(1, 0.25, n),
            "cargo_decisor": rng.binomial(1, 0.35, n),
            "tamano_empresa": rng.choice(["pequena", "mediana", "grande"], n, p=[0.5, 0.35, 0.15]),
            "sector": rng.choice(["manufactura", "servicios", "comercio", "tecnologia"], n),
            "fuente": rng.choice(["organico", "pauta", "referido", "evento"], n, p=[0.4, 0.3, 0.15, 0.15]),
        }
    )
    # Probabilidad "real" de conversión, desconocida para el modelo.
    logit = (
        -4.2
        + 0.18 * df.visitas_web
        + 0.05 * df.paginas_vistas
        + 0.55 * df.descargas_contenido
        + 0.10 * df.emails_abiertos
        + 0.35 * df.clics_email
        - 0.025 * df.dias_ultima_interaccion
        + 1.30 * df.lleno_formulario
        + 0.90 * df.cargo_decisor
        + df.tamano_empresa.map({"pequena": 0.0, "mediana": 0.4, "grande": 0.8})
        + df.fuente.map({"organico": 0.0, "pauta": -0.2, "referido": 0.9, "evento": 0.5})
    )
    df["convirtio"] = rng.binomial(1, 1 / (1 + np.exp(-logit)))
    df.insert(0, "id_lead", [f"L{i:05d}" for i in range(n)])
    return df


def construir_modelo() -> Pipeline:
    prepro = ColumnTransformer(
        [
            ("num", StandardScaler(), NUMERICAS),
            ("bin", "passthrough", BINARIAS),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAS),
        ]
    )
    return Pipeline([("prepro", prepro), ("clf", LogisticRegression(max_iter=1000))])


def calificar(modelo: Pipeline, leads: pd.DataFrame) -> pd.DataFrame:
    out = leads.copy()
    out["puntaje"] = (modelo.predict_proba(out[NUMERICAS + BINARIAS + CATEGORICAS])[:, 1] * 100).round(1)
    # Un SQL necesita puntaje alto y una señal explícita de intención o autoridad.
    es_sql = (out.puntaje >= UMBRAL_SQL) & ((out.lleno_formulario == 1) | (out.cargo_decisor == 1))
    es_mql = out.puntaje >= UMBRAL_MQL
    out["etapa"] = np.select([es_sql, es_mql], ["SQL", "MQL"], default="Nutricion")
    return out


def main() -> None:
    datos = generar_leads()
    X = datos[NUMERICAS + BINARIAS + CATEGORICAS]
    y = datos.convirtio
    X_tr, X_te, y_tr, y_te, id_tr, id_te = train_test_split(
        X, y, datos.id_lead, test_size=0.3, random_state=0, stratify=y
    )

    modelo = construir_modelo().fit(X_tr, y_tr)
    auc = roc_auc_score(y_te, modelo.predict_proba(X_te)[:, 1])

    prueba = calificar(modelo, datos[datos.id_lead.isin(id_te)])
    resumen = (
        prueba.groupby("etapa")
        .agg(leads=("id_lead", "count"), tasa_conversion=("convirtio", "mean"))
        .reindex(["SQL", "MQL", "Nutricion"])
    )
    resumen["tasa_conversion"] = (resumen.tasa_conversion * 100).round(1)

    # Pesos del modelo: qué señales suben o bajan el puntaje.
    nombres = modelo.named_steps["prepro"].get_feature_names_out()
    pesos = pd.Series(modelo.named_steps["clf"].coef_[0], index=nombres).sort_values()

    # Gráfica: tasa de conversión real por decil de puntaje.
    prueba["decil"] = pd.qcut(prueba.puntaje, 10, labels=False, duplicates="drop") + 1
    por_decil = prueba.groupby("decil").convirtio.mean() * 100
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(por_decil.index, por_decil.values, color="#2f6f8f")
    ax.set_xlabel("Decil de puntaje (1 = más bajo, 10 = más alto)")
    ax.set_ylabel("Conversión real (%)")
    ax.set_title(f"Validación del lead scoring (AUC = {auc:.2f})")
    ax.set_xticks(por_decil.index)
    fig.tight_layout()
    fig.savefig(SALIDA / "conversion_por_decil.png", dpi=150)

    calificar(modelo, datos).to_csv(SALIDA / "leads_calificados.csv", index=False)
    joblib.dump(modelo, SALIDA / "modelo_lead_scoring.joblib")

    print(f"AUC en prueba: {auc:.3f}\n")
    print("Conversión real por etapa (datos de prueba):")
    print(resumen.to_string(), "\n")
    print("Pesos del modelo (variables estandarizadas):")
    print(pesos.round(2).to_string())


if __name__ == "__main__":
    main()
