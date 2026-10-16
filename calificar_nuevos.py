"""
Califica automáticamente leads nuevos con el modelo ya entrenado.

Pensado para conectarse a un flujo de automatización: el CSV de entrada puede ser
la exportación de un formulario de WordPress o de un CRM, y el CSV de salida
trae el puntaje y la etapa (SQL / MQL / Nutricion) lista para reimportar o
para disparar una acción (asignar a ventas, enviar a una secuencia de correos).

Uso:
    python calificar_nuevos.py nuevos_leads.csv
"""

import sys
from pathlib import Path

import joblib
import pandas as pd

from lead_scoring import SALIDA, calificar


def main(ruta_entrada: str) -> None:
    modelo_path = SALIDA / "modelo_lead_scoring.joblib"
    if not modelo_path.exists():
        sys.exit("Primero entrena el modelo con: python lead_scoring.py")

    modelo = joblib.load(modelo_path)
    nuevos = pd.read_csv(ruta_entrada)
    calificados = calificar(modelo, nuevos).sort_values("puntaje", ascending=False)

    salida = Path(ruta_entrada).with_name(Path(ruta_entrada).stem + "_calificados.csv")
    calificados.to_csv(salida, index=False)

    print(calificados[["id_lead", "puntaje", "etapa"]].to_string(index=False))
    print(f"\nResultado guardado en {salida}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "nuevos_leads.csv")
