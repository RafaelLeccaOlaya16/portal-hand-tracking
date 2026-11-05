
import os
import time
import urllib.request

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hand_landmarker.task")
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/latest/hand_landmarker.task"
)


def asegurar_modelo():
    if not os.path.exists(MODEL_PATH):
        print("Descargando modelo de deteccion de manos (una sola vez)...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Modelo descargado:", MODEL_PATH)

def filtro_sketch(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    inv = 255 - gray
    blur = cv2.GaussianBlur(inv, (21, 21), 0)
    sketch_gray = cv2.divide(gray, 255 - blur, scale=256)
    return cv2.cvtColor(sketch_gray, cv2.COLOR_GRAY2BGR)


def filtro_pixelate(frame, bloques=32):
    h, w = frame.shape[:2]
    temp = cv2.resize(frame, (bloques, int(bloques * h / w)), interpolation=cv2.INTER_LINEAR)
    return cv2.resize(temp, (w, h), interpolation=cv2.INTER_NEAREST)


def filtro_invertido(frame):
    return 255 - frame


def filtro_termico(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.applyColorMap(gray, cv2.COLORMAP_JET)


def filtro_cartoon(frame):
    suavizado = cv2.bilateralFilter(frame, d=9, sigmaColor=150, sigmaSpace=150)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.medianBlur(gray, 5)
    bordes = cv2.adaptiveThreshold(
        gray_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 6
    )
    bordes_bgr = cv2.cvtColor(bordes, cv2.COLOR_GRAY2BGR)
    return cv2.bitwise_and(suavizado, bordes_bgr)


def filtro_bordes(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    bordes = cv2.Canny(gray, 80, 160)
    return cv2.cvtColor(bordes, cv2.COLOR_GRAY2BGR)


FILTROS = {
    "s": filtro_sketch,
    "p": filtro_pixelate,
    "i": filtro_invertido,
    "t": filtro_termico,
    "c": filtro_cartoon,
    "e": filtro_bordes,
}
ORDEN_FILTROS = list(FILTROS.keys()) 

THUMB_TIP = 4
INDEX_TIP = 8
PINKY_TIP = 20
WRIST = 0
MIDDLE_MCP = 9


def obtener_puntos_portal(resultado, w, h):
    if not resultado.hand_landmarks or len(resultado.hand_landmarks) < 2:
        return None

    puntos = []
    for mano in resultado.hand_landmarks[:2]:
        for idx in (THUMB_TIP, INDEX_TIP):
            lm = mano[idx]
            puntos.append((int(lm.x * w), int(lm.y * h)))

    puntos = np.array(puntos, dtype=np.int32)
    centro = puntos.mean(axis=0)
    angulos = np.arctan2(puntos[:, 1] - centro[1], puntos[:, 0] - centro[0])
    orden = np.argsort(angulos)
    return puntos[orden]


def pulgar_toca_menique(mano):
    pulgar = mano[THUMB_TIP]
    menique = mano[PINKY_TIP]
    muneca = mano[WRIST]
    medio_mcp = mano[MIDDLE_MCP]

    dist_dedos = ((pulgar.x - menique.x) ** 2 + (pulgar.y - menique.y) ** 2) ** 0.5
    tam_mano = ((muneca.x - medio_mcp.x) ** 2 + (muneca.y - medio_mcp.y) ** 2) ** 0.5
    if tam_mano == 0:
        return False
    return (dist_dedos / tam_mano) < 0.55

def main():
    asegurar_modelo()

    base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
    options = mp_vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.6,
        min_tracking_confidence=0.6,
    )
    landmarker = mp_vision.HandLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("No se pudo abrir la camara.")
        return

    indice_filtro = 0
    filtro_actual = ORDEN_FILTROS[indice_filtro]
    inicio = time.time()

    gesto_activo_prev = False
    ultimo_cambio = 0.0
    cooldown_gesto = 0.5 

    print("Portal Python corriendo.")
    print("Junta pulgar + meñique para cambiar de filtro, o usa el teclado.")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp_ms = int((time.time() - inicio) * 1000)
        resultado = landmarker.detect_for_video(mp_image, timestamp_ms)

        salida = frame.copy()
        puntos = obtener_puntos_portal(resultado, w, h)

        gesto_activo = False
        if resultado.hand_landmarks:
            for mano in resultado.hand_landmarks:
                if pulgar_toca_menique(mano):
                    gesto_activo = True
                    break

        ahora = time.time()
        if gesto_activo and not gesto_activo_prev and (ahora - ultimo_cambio) > cooldown_gesto:
            indice_filtro = (indice_filtro + 1) % len(ORDEN_FILTROS)
            filtro_actual = ORDEN_FILTROS[indice_filtro]
            ultimo_cambio = ahora
        gesto_activo_prev = gesto_activo

        if puntos is not None and filtro_actual in FILTROS:
            frame_filtrado = FILTROS[filtro_actual](frame)

            mascara = np.zeros((h, w), dtype=np.uint8)
            cv2.fillConvexPoly(mascara, puntos, 255)

            mascara_3c = cv2.merge([mascara, mascara, mascara])
            salida = np.where(mascara_3c == 255, frame_filtrado, salida)

            cv2.polylines(salida, [puntos], isClosed=True, color=(255, 255, 255), thickness=2)

        color_aviso = (0, 255, 0) if not gesto_activo else (0, 165, 255)
        cv2.putText(salida, f"Filtro: {filtro_actual}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color_aviso, 2)

        cv2.imshow("Portal Python", salida)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("n"):
            filtro_actual = "n"
        elif chr(key) in FILTROS:
            filtro_actual = chr(key)
            indice_filtro = ORDEN_FILTROS.index(filtro_actual)

    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()


if __name__ == "__main__":
    main()