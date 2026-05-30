import numpy as np
import pickle
import os
from datetime import datetime
from typing import Optional

# Ventana de tiempo del modelo
VENTANA = 24

# Umbrales de distancia en cm
DIST_PRECAUCION = 200.0
DIST_ALERTA     = 100.0
DIST_EMERGENCIA =  50.0

# Rutas de los archivos del modelo
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "modelo_lstm_mopan.keras")
SCALER_X   = os.path.join(BASE_DIR, "scaler_X_mopan.pkl")
SCALER_Y   = os.path.join(BASE_DIR, "scaler_y_mopan.pkl")

# Carga lazy del modelo
_modelo   = None
_scaler_X = None
_scaler_y = None

def cargar_modelo():
    global _modelo, _scaler_X, _scaler_y
    if _modelo is None:
        try:
            import tensorflow as tf
            _modelo = tf.keras.models.load_model(MODEL_PATH)
            with open(SCALER_X, 'rb') as f:
                _scaler_X = pickle.load(f)
            with open(SCALER_Y, 'rb') as f:
                _scaler_y = pickle.load(f)
            print("[LSTM] Modelo cargado correctamente")
        except Exception as e:
            print(f"[LSTM] Error al cargar modelo: {e}")
            _modelo = None
    return _modelo is not None


def predecir_lstm(mediciones: list) -> Optional[dict]:
    """
    Recibe las últimas mediciones de la BD y predice
    los minutos hasta que la distancia baje de DIST_PRECAUCION.

    Features del modelo (en orden):
    nivel_cm, voltaje_bateria, epoch_lluviosa, hora_dia, mes
    """
    if not cargar_modelo():
        return None

    # Filtrar errores de sensor
    validas = [m for m in mediciones if m.nivel_cm < 900]

    if len(validas) < VENTANA:
        return {
            "disponible": False,
            "mensaje": f"Acumulando datos para LSTM ({len(validas)}/{VENTANA} mediciones)",
            "n_mediciones": len(validas)
        }

    # Tomar las últimas VENTANA mediciones
    recientes = validas[-VENTANA:]

    # Construir matriz de features
    X = []
    for m in recientes:
        ts = m.timestamp
        hora_dia     = ts.hour
        mes          = ts.month
        epoch_lluvia = 1 if mes in [5, 6, 7, 8, 9, 10] else 0  # Mayo-Octubre = lluvia

        X.append([
            m.nivel_cm,
            m.voltaje_bateria,
            epoch_lluvia,
            hora_dia,
            mes
        ])

    X = np.array(X, dtype=np.float32)

    # Normalizar
    X_scaled = _scaler_X.transform(X)

    # Reshape para LSTM: (1, ventana, features)
    X_input = X_scaled.reshape(1, VENTANA, 5)

    # Predecir
    y_pred_scaled = _modelo.predict(X_input, verbose=0)
    minutos_pred  = float(_scaler_y.inverse_transform(y_pred_scaled)[0][0])
    minutos_pred  = max(0, round(minutos_pred))

    # Calcular tasa de cambio actual
    dist_actual = recientes[-1].nivel_cm
    dist_prev   = recientes[-2].nivel_cm
    dt_min = (recientes[-1].timestamp - recientes[-2].timestamp).total_seconds() / 60
    tasa   = (dist_actual - dist_prev) / dt_min if dt_min > 0 else 0.0

    # Determinar estado
    if dist_actual <= DIST_EMERGENCIA:
        estado = "EMERGENCIA"
    elif dist_actual <= DIST_ALERTA:
        estado = "ALERTA"
    elif dist_actual <= DIST_PRECAUCION:
        estado = "PRECAUCION"
    else:
        estado = "NORMAL"

    # Interpretación
    if estado == "EMERGENCIA":
        interpretacion = f"EMERGENCIA ACTIVA. Agua a {dist_actual:.0f}cm del sensor."
    elif estado == "ALERTA":
        interpretacion = f"Zona de alerta. Agua a {dist_actual:.0f}cm. LSTM estima {minutos_pred} min a precaución."
    elif tasa < -0.5 and minutos_pred > 0:
        interpretacion = f"Río subiendo a {abs(tasa):.2f} cm/min. LSTM predice precaución en {minutos_pred} minutos."
    elif tasa > 0.5:
        interpretacion = f"Río bajando ({tasa:.2f} cm/min). Sin riesgo inmediato."
    else:
        interpretacion = f"Nivel estable en {dist_actual:.0f}cm. LSTM: {minutos_pred} min a precaución."

    return {
        "disponible":          True,
        "dist_actual_cm":      round(dist_actual, 1),
        "tasa_cambio_cm_min":  round(tasa, 3),
        "estado":              estado,
        "minutos_a_precaucion": minutos_pred,
        "interpretacion":      interpretacion,
        "ventana_usada":       VENTANA,
        "n_mediciones":        len(validas),
        "modelo":              "LSTM (2 capas, 24 pasos)"
    }
