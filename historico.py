import pandas as pd
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "dataset_rio_mopan.csv")

# En el CSV nivel_cm es el NIVEL DEL RIO (mayor = mas peligro)
# Umbrales del CSV
CSV_PREC = 120.0
CSV_ALERT = 160.0
CSV_EMER  = 200.0

_df = None

def cargar_dataset():
    global _df
    if _df is None:
        try:
            _df = pd.read_csv(CSV_PATH)
            _df['timestamp'] = pd.to_datetime(_df['timestamp'])
            _df = _df.sort_values('timestamp').reset_index(drop=True)
            _df['tasa'] = _df['nivel_cm'].diff() / 5
            print(f"[HISTORICO] Dataset cargado: {len(_df)} filas")
        except Exception as e:
            print(f"[HISTORICO] Error: {e}")
            _df = None
    return _df is not None


def comparar_historico(dist_actual: float, tasa_actual: float) -> dict:
    if not cargar_dataset():
        return {"disponible": False, "mensaje": "Dataset histórico no disponible"}

    MESES = {1:'Enero',2:'Febrero',3:'Marzo',4:'Abril',5:'Mayo',6:'Junio',
             7:'Julio',8:'Agosto',9:'Septiembre',10:'Octubre',11:'Noviembre',12:'Diciembre'}

    # Convertir distancia actual a nivel equivalente del CSV
    # dist_actual: 300cm = normal, 50cm = emergencia
    # nivel CSV:   30cm  = normal, 200cm = emergencia
    # Mapeo inverso: nivel_csv = 300 - dist_actual (aproximado)
    nivel_equiv = max(30, min(245, 300 - dist_actual))

    # Determinar estado actual
    if dist_actual <= 50:
        estado_actual = "EMERGENCIA"
        # Buscar episodios en zona de alerta/emergencia del CSV
        similares = _df[(_df['nivel_cm'] >= CSV_ALERT)].copy()
        # Calcular cuánto tardó en volver a normal
        resultados = []
        for idx in similares.index[:200]:
            futuro = _df.loc[idx:min(idx+500, len(_df)-1)]
            vuelta = futuro[futuro['nivel_cm'] < CSV_PREC]
            if len(vuelta) > 0:
                dur = (vuelta.iloc[0]['timestamp'] - _df.loc[idx,'timestamp']).total_seconds() / 60
                if dur > 0:
                    resultados.append({'duracion_min': round(dur), 'mes': int(_df.loc[idx,'timestamp'].month)})
        label = "duración en zona de alerta/emergencia antes de volver a normal"

    elif dist_actual <= 100:
        estado_actual = "ALERTA"
        similares = _df[(_df['nivel_cm'] >= CSV_ALERT) & (_df['nivel_cm'] < CSV_EMER)].copy()
        resultados = []
        for idx in similares.index[:200]:
            futuro = _df.loc[idx:min(idx+500, len(_df)-1)]
            vuelta = futuro[futuro['nivel_cm'] < CSV_PREC]
            if len(vuelta) > 0:
                dur = (vuelta.iloc[0]['timestamp'] - _df.loc[idx,'timestamp']).total_seconds() / 60
                if dur > 0:
                    resultados.append({'duracion_min': round(dur), 'mes': int(_df.loc[idx,'timestamp'].month)})
        label = "duración en zona de alerta antes de volver a normal"

    elif dist_actual <= 200:
        estado_actual = "PRECAUCION"
        similares = _df[(_df['nivel_cm'] >= CSV_PREC) & (_df['nivel_cm'] < CSV_ALERT) & (_df['tasa'] > 0.1)].copy()
        resultados = []
        for idx in similares.index[:200]:
            futuro = _df.loc[idx:min(idx+500, len(_df)-1)]
            llegada = futuro[futuro['nivel_cm'] >= CSV_ALERT]
            if len(llegada) > 0:
                dur = (llegada.iloc[0]['timestamp'] - _df.loc[idx,'timestamp']).total_seconds() / 60
                if dur > 0:
                    resultados.append({'duracion_min': round(dur), 'mes': int(_df.loc[idx,'timestamp'].month)})
        label = "tiempo en precaución antes de llegar a alerta"

    else:
        estado_actual = "NORMAL"
        similares = _df[(_df['nivel_cm'] >= 80) & (_df['nivel_cm'] < CSV_PREC) & (_df['tasa'] > 0.1)].copy()
        resultados = []
        for idx in similares.index[:200]:
            futuro = _df.loc[idx:min(idx+500, len(_df)-1)]
            llegada = futuro[futuro['nivel_cm'] >= CSV_PREC]
            if len(llegada) > 0:
                dur = (llegada.iloc[0]['timestamp'] - _df.loc[idx,'timestamp']).total_seconds() / 60
                if dur > 0:
                    resultados.append({'duracion_min': round(dur), 'mes': int(_df.loc[idx,'timestamp'].month)})
        label = "tiempo hasta alcanzar precaución"

    if not resultados:
        return {
            "disponible":   False,
            "mensaje":      f"Sin episodios históricos en estado {estado_actual}",
            "n_similares":  len(similares)
        }

    mins    = [r['duracion_min'] for r in resultados]
    meses   = [r['mes'] for r in resultados]
    mes_fq  = max(set(meses), key=meses.count)

    return {
        "disponible":         True,
        "estado_actual":      estado_actual,
        "dist_referencia_cm": dist_actual,
        "n_episodios":        len(resultados),
        "min_promedio":       round(np.mean(mins)),
        "min_mediana":        round(np.median(mins)),
        "peor_caso_min":      max(mins),
        "mejor_caso_min":     min(mins),
        "mes_mas_comun":      MESES.get(mes_fq, ''),
        "tipo_comparacion":   label,
        "interpretacion": (
            f"En {len(resultados)} episodios históricos en estado {estado_actual}, "
            f"el promedio de {label} fue {round(np.mean(mins))} minutos. "
            f"Caso más rápido: {min(mins)} min. Caso más lento: {max(mins)} min. "
            f"Mes más frecuente: {MESES.get(mes_fq, '')}."
        )
    }
