"""
Detección de vehículos con YOLO, detección de matrículas con YOLO (modelo específico)
y lectura del texto con EasyOCR (sin dependencia de cuDNN).
Flujo: 1) YOLO coches → 2) YOLO matrículas dentro del coche → 3) OCR en el crop de la placa.
"""

import argparse
import re
import sys
from pathlib import Path

import cv2
import easyocr
from ultralytics import YOLO

# Dispositivo para inferencia: se establece en cargar_modelos(); usado por EasyOCR (lazy)
_use_gpu_ocr = False

EXTENSIONES_VIDEO = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v", ".wmv"}
EXTENSIONES_IMAGEN = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"}

# Modelo YOLO de matrículas (Hugging Face). Nano = rápido; usar v1s/v1m si necesitas más precisión.
REPO_PLACAS_HF = "morsetechlab/yolov11-license-plate-detection"
ARCHIVO_PLACAS_HF = "license-plate-finetune-v1n.pt"
# Ruta local: si existe este .pt no se descarga de HF
NOMBRE_LOCAL_PLACAS = "license-plate-finetune-v1n.pt"

# Nombres de los modelos exportados a ONNX (generados por export_engine.py)
NOMBRE_ONNX_COCHES = "yolo11n.onnx"
NOMBRE_ONNX_PLACAS = "license-plate-finetune-v1n.onnx"

# Clases COCO: 2=car, 3=motorcycle, 5=bus, 7=truck
CLASES_VEHICULOS = (2, 3, 5, 7)

# Patrón para matrículas: números y letras (español: 1234 ABC o 1234ABC); longitud 3-12
PATRON_MATRICULA = re.compile(r"^[A-Z0-9]{3,12}$", re.IGNORECASE)


def limpiar_texto_matricula(texto: str) -> str:
    """Elimina espacios y caracteres que no son alfanuméricos."""
    if not texto or not isinstance(texto, str):
        return ""
    return re.sub(r"[\s\-\.]", "", texto.strip().upper())


def parece_matricula(texto: str) -> bool:
    """Comprueba si el texto tiene formato de matrícula (alfanumérico, longitud típica)."""
    limpio = limpiar_texto_matricula(texto)
    if len(limpio) < 3 or len(limpio) > 12:
        return False
    return bool(PATRON_MATRICULA.match(limpio))


def _obtener_ruta_modelo_placas() -> Path:
    """Devuelve la ruta al .pt de detección de matrículas; lo descarga de HF si no existe."""
    local = Path(__file__).resolve().parent / NOMBRE_LOCAL_PLACAS
    if local.is_file():
        return local
    try:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(
            repo_id=REPO_PLACAS_HF,
            filename=ARCHIVO_PLACAS_HF,
            local_dir=Path(__file__).resolve().parent,
            local_dir_use_symlinks=False,
        )
        return Path(path)
    except Exception:
        return local  # fallback: intentará cargar y fallará con mensaje claro


def _get_device():
    """Devuelve 'cuda:0' si hay GPU disponible, si no 'cpu'."""
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda:0"
    except Exception:
        pass
    return "cpu"


def _resolver_modelo(base_dir, nombre_pt, nombre_onnx, ruta_override=None):
    """
    Resuelve qué archivo de modelo cargar con esta prioridad:
      1. .pt  → PyTorch con CUDA (funciona en sm_61 con cu118, es la opción óptima)
      2. .onnx → onnxruntime (requiere cuDNN instalado, omitido si falta)
      3. .engine → TensorRT (NO compatible con GTX 1050 Ti sm_61, ignorado)
    Devuelve (ruta: Path, usar_device: bool).
    """
    if ruta_override:
        ruta_pt = Path(ruta_override)
    else:
        ruta_pt = base_dir / nombre_pt

    # .pt con PyTorch+CUDA es la opción más estable para sm_61 + cu118
    if ruta_pt.is_file():
        return ruta_pt, True

    # Fallback a .onnx solo si no hay .pt (requiere cuDNN)
    ruta_onnx = base_dir / nombre_onnx
    if ruta_onnx.is_file():
        return ruta_onnx, True

    # .engine no soportado en sm_61, pero lo intentamos como último recurso
    ruta_engine = ruta_pt.with_suffix(".engine")
    if ruta_engine.is_file():
        return ruta_engine, False

    return ruta_pt, True  # deja que falle con mensaje claro


def cargar_modelos(yolo_coches: str = "yolo11n.pt", yolo_placas=None):
    """
    Carga: YOLO coches, YOLO matrículas (modelo específico) y EasyOCR.

    Prioridad de modelos (de mayor a menor rendimiento para GTX 1050 Ti):
      1. .onnx  → onnxruntime-gpu, funciona en cualquier GPU incluyendo Pascal sm_61
      2. .engine → TensorRT (NO soportado en GTX 1050 Ti sm_61, se ignora)
      3. .pt    → PyTorch, fallback universal

    Para generar los .onnx: python export_engine.py --formato onnx --half
    EasyOCR en CPU (no requiere cuDNN del sistema).
    """
    global _use_gpu_ocr
    device = _get_device()
    _use_gpu_ocr = device != "cpu"

    base_dir = Path(__file__).resolve().parent
    argv_orig = list(sys.argv)
    sys.argv = ["python"]
    try:
        # --- Modelo coches ---
        ruta_coches, usar_device_coches = _resolver_modelo(
            base_dir, yolo_coches, NOMBRE_ONNX_COCHES
        )
        if ruta_coches.suffix == ".pt" and not ruta_coches.is_absolute():
            modelo_coches = YOLO(yolo_coches)  # Ultralytics resuelve y descarga si hace falta
        else:
            modelo_coches = YOLO(str(ruta_coches))
        modelo_coches._use_device = device if usar_device_coches else None
        print(f"  Modelo coches:     {ruta_coches.name}  [device={modelo_coches._use_device or 'engine'}]")

        # --- Modelo matrículas ---
        ruta_placas_base = Path(yolo_placas) if yolo_placas else _obtener_ruta_modelo_placas()
        ruta_placas, usar_device_placas = _resolver_modelo(
            base_dir,
            ruta_placas_base.name,
            NOMBRE_ONNX_PLACAS,
            ruta_override=yolo_placas,
        )
        if not ruta_placas.is_file():
            raise FileNotFoundError(
                f"Modelo de matrículas no encontrado: {ruta_placas}. "
                f"Descarga manual: https://huggingface.co/{REPO_PLACAS_HF} (archivo {ARCHIVO_PLACAS_HF})"
            )
        modelo_placas = YOLO(str(ruta_placas))
        modelo_placas._use_device = device if usar_device_placas else None
        print(f"  Modelo matrículas: {ruta_placas.name}  [device={modelo_placas._use_device or 'engine'}]")

        # --- EasyOCR: no necesita cuDNN, funciona en CPU y GPU ---
        # gpu=False: evita dependencia de cuDNN del sistema (no instalado)
        print("  Cargando EasyOCR (CPU)...")
        ocr = easyocr.Reader(["en"], gpu=False, verbose=False)
        print("  EasyOCR listo.")
        return modelo_coches, modelo_placas, ocr
    finally:
        sys.argv[:] = argv_orig


# EasyOCR es ahora el OCR principal (PaddleOCR eliminado: requería cuDNN no instalado)


def detectar_placa_en_crop(modelo_placas, imagen_coche, conf_min=0.25):
    """
    Ejecuta el modelo YOLO de matrículas sobre el recorte del coche.
    Devuelve la mejor (bbox en coords del crop, confianza) o (None, None).
    bbox = (x1, y1, x2, y2) en coordenadas de imagen_coche.
    """
    if imagen_coche is None or imagen_coche.size == 0:
        return None, None
    device = getattr(modelo_placas, "_use_device", None)
    kwargs = {"conf": conf_min, "verbose": False}
    if device is not None:
        kwargs["device"] = device
    try:
        res = modelo_placas(imagen_coche, **kwargs)[0]
    except Exception:
        return None, None
    if res.boxes is None or len(res.boxes) == 0:
        return None, None
    # Quedarnos con la detección de mayor confianza (clase 0 = license_plate)
    best = None
    best_conf = 0.0
    for box in res.boxes:
        c = float(box.conf[0])
        if c > best_conf:
            best_conf = c
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            best = (x1, y1, x2, y2)
    if best is None:
        return None, None
    return best, best_conf


def extraer_zona_busqueda(imagen, bbox, usar_coche_completo=True):
    """
    Recorta la zona donde buscar la matrícula.
    Si usar_coche_completo=True (recomendado): usa todo el bbox del coche y deja que
    el OCR encuentre la placa en cualquier posición (delantera, trasera, ángulo).
    Si False: solo la mitad inferior del coche (comportamiento antiguo).
    Devuelve (zona_imagen, (ox, oy)) con origen de la zona en el frame.
    """
    h, w = imagen.shape[:2]
    x1, y1, x2, y2 = map(int, bbox)
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w, x2)
    y2 = min(h, y2)
    if usar_coche_completo:
        zona = imagen[y1:y2, x1:x2].copy()
        return zona, (x1, y1)
    # Fallback: solo mitad inferior
    alto = y2 - y1
    y_inicio = y1 + (alto // 2)
    zona = imagen[y_inicio:y2, x1:x2].copy()
    return zona, (x1, y_inicio)


def _cuadrilatero_a_bbox(box):
    """Convierte caja de 4 puntos [[x,y],...] a (x1, y1, x2, y2) axis-aligned."""
    xs = [p[0] for p in box]
    ys = [p[1] for p in box]
    return (min(xs), min(ys), max(xs), max(ys))


def ocr_buscar_matricula_y_caja(ocr, imagen_zona):
    """
    Ejecuta EasyOCR en la zona y devuelve (texto_matricula, bbox_en_zona).
    Elige el texto que más se parece a una matrícula por aspecto, confianza y posición.
    ocr: instancia de easyocr.Reader
    """
    if imagen_zona is None or imagen_zona.size == 0:
        return None, None
    try:
        resultados = ocr.readtext(imagen_zona)
    except Exception:
        return None, None
    if not resultados:
        return None, None
    zh, zw = imagen_zona.shape[:2]
    candidatos = []
    for (box_pts, texto, conf) in resultados:
        if not isinstance(texto, str) or not texto.strip():
            continue
        limpio = limpiar_texto_matricula(texto)
        if not limpio or not parece_matricula(limpio):
            continue
        try:
            # EasyOCR devuelve box como lista de 4 puntos [[x,y],...]
            bbox = _cuadrilatero_a_bbox(box_pts)
        except (IndexError, TypeError):
            continue
        bw = bbox[2] - bbox[0]
        bh = bbox[3] - bbox[1]
        if bh <= 0:
            continue
        aspecto = bw / bh
        score = conf * 10 + (3.0 if 1.5 <= aspecto <= 8 else 0.0)
        cy = (bbox[1] + bbox[3]) / 2
        if zh > 0 and cy > zh * 0.4:
            score += 1.5
        candidatos.append((limpio, bbox, score, conf))
    if not candidatos:
        # Fallback: cualquier texto alfanumérico >= 4 chars
        for (box_pts, texto, conf) in resultados:
            if not isinstance(texto, str) or len(texto.strip()) < 4:
                continue
            limpio = limpiar_texto_matricula(texto)
            if len(limpio) >= 4 and re.match(r"^[A-Z0-9]+$", limpio, re.I):
                try:
                    bbox = _cuadrilatero_a_bbox(box_pts)
                    candidatos.append((limpio, bbox, 1.0, 0.5))
                except (IndexError, TypeError):
                    pass
    if not candidatos:
        return None, None
    candidatos.sort(key=lambda x: -x[2])
    return candidatos[0][0], candidatos[0][1]


# Altura mínima del crop de placa para OCR (crops muy pequeños se leen mal)
ALTURA_MIN_PLACA_OCR = 40


def preprocesar_placa_para_ocr(imagen):
    """
    Redimensiona y mejora el contraste del crop de la placa para que el OCR lea mejor.
    Devuelve imagen BGR.
    """
    if imagen is None or imagen.size == 0:
        return imagen
    h, w = imagen.shape[:2]
    if h < ALTURA_MIN_PLACA_OCR and h > 0:
        escala = ALTURA_MIN_PLACA_OCR / h
        nw = max(1, int(w * escala))
        nh = ALTURA_MIN_PLACA_OCR
        imagen = cv2.resize(imagen, (nw, nh), interpolation=cv2.INTER_CUBIC)
    # Mejorar contraste (útil para placas con poco contraste)
    gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    gris = cv2.normalize(gris, None, 0, 255, cv2.NORM_MINMAX)
    # Volver a BGR (3 canales para EasyOCR)
    return cv2.cvtColor(gris, cv2.COLOR_GRAY2BGR)


def _extraer_texto_matricula_de_resultados_ocr(resultados):
    """Extrae el mejor candidato a matrícula de la lista de resultados de PaddleOCR (función legacy, no usada)."""
    if not resultados or not resultados[0]:
        return None
    candidatos = []
    for linea in resultados[0]:
        if not linea or len(linea) < 2:
            continue
        texto = linea[1][0] if isinstance(linea[1], (list, tuple)) else linea[1]
        conf = linea[1][1] if isinstance(linea[1], (list, tuple)) and len(linea[1]) > 1 else 0.0
        if not isinstance(texto, str) or not texto.strip():
            continue
        limpio = limpiar_texto_matricula(texto)
        if limpio and parece_matricula(limpio):
            candidatos.append((limpio, conf))
    if not candidatos:
        for linea in resultados[0]:
            if not linea or len(linea) < 2:
                continue
            texto = linea[1][0] if isinstance(linea[1], (list, tuple)) else linea[1]
            if not isinstance(texto, str):
                continue
            limpio = limpiar_texto_matricula(texto)
            if len(limpio) >= 2 and re.match(r"^[A-Z0-9]+$", limpio, re.I):
                conf = linea[1][1] if isinstance(linea[1], (list, tuple)) and len(linea[1]) > 1 else 0.5
                candidatos.append((limpio, conf))
    if not candidatos:
        return None
    candidatos.sort(key=lambda x: (-len(x[0]), -x[1]))
    return candidatos[0][0]


def _extraer_matricula_de_easyocr(resultados_easyocr):
    """De la lista [(bbox, texto, conf), ...] de EasyOCR devuelve el mejor candidato a matrícula."""
    if not resultados_easyocr:
        return None
    candidatos = []
    for _box, texto, conf in resultados_easyocr:
        if not isinstance(texto, str) or not texto.strip():
            continue
        limpio = limpiar_texto_matricula(texto)
        if limpio and parece_matricula(limpio):
            candidatos.append((limpio, conf))
    if not candidatos:
        for _box, texto, conf in resultados_easyocr:
            if not isinstance(texto, str):
                continue
            limpio = limpiar_texto_matricula(texto)
            if len(limpio) >= 2 and re.match(r"^[A-Z0-9]+$", limpio, re.I):
                candidatos.append((limpio, conf))
    if not candidatos:
        return None
    candidatos.sort(key=lambda x: (-len(x[0]), -x[1]))
    return candidatos[0][0]


def leer_matricula_en_imagen(ocr, imagen):
    """
    Lee la matrícula en el crop usando EasyOCR.
    Intenta primero con imagen preprocesada (mejor contraste), luego con la original.
    ocr: instancia de easyocr.Reader
    """
    if imagen is None or imagen.size == 0:
        return None

    img_prep = preprocesar_placa_para_ocr(imagen)

    # 1) EasyOCR con imagen preprocesada
    for img in (img_prep, imagen):
        if img is None or img.size == 0:
            continue
        try:
            res = ocr.readtext(img)
            texto = _extraer_matricula_de_easyocr(res)
            if texto:
                return texto
        except Exception:
            pass
    return None


def _procesar_una_caja(imagen, box, resultados_yolo, modelo_placas, ocr, conf_min_para_matricula, track_id=None, escala=None):
    """
    Dado un box de YOLO (coche), extrae zona, detecta placa, OCR. Devuelve dict o None.
    track_id: opcional, para tracking (int).
    escala: (escala_x, escala_y) para reescalar coords si el frame fue reducido antes de YOLO.
    """
    cls_id = int(box.cls[0])
    if cls_id not in CLASES_VEHICULOS:
        return None

    conf = float(box.conf[0])
    x1, y1, x2, y2 = box.xyxy[0].tolist()
    if escala is not None:
        sx, sy = escala
        x1, x2 = x1 * sx, x2 * sx
        y1, y2 = y1 * sy, y2 * sy

    # Zona donde buscar la matrícula (misma lógica de siempre)
    zona, (ox, oy) = extraer_zona_busqueda(imagen, (x1, y1, x2, y2))
    zh, zw = zona.shape[:2]

    nombre_clase = resultados_yolo.names.get(cls_id, f"clase_{cls_id}")

    # Resultado base: siempre devolvemos el coche con su bbox, aunque no se lea matrícula
    out_base = {
        "bbox": (x1, y1, x2, y2),
        "clase": nombre_clase,
        "confianza": conf,
        "matricula": None,
        "plate_bbox": None,
        "plate_crop": None,
        "plate_conf": None,
    }
    if track_id is not None:
        out_base["track_id"] = track_id

    # Si el crop es demasiado pequeño, no intentamos OCR pero sí devolvemos el coche
    if zh < 10 or zw < 10:
        return out_base

    # Solo intentamos detectar/leer matrícula si la confianza del coche es suficiente
    if conf < conf_min_para_matricula:
        return out_base

    plate_bbox_crop, plate_conf = detectar_placa_en_crop(modelo_placas, zona, conf_min=0.25)
    if plate_bbox_crop is not None:
        px1, py1, px2, py2 = [int(v) for v in plate_bbox_crop]
        px1, py1 = max(0, px1), max(0, py1)
        px2, py2 = min(zw, px2), min(zh, py2)
        if px2 > px1 and py2 > py1:
            plate_crop = zona[py1:py2, px1:px2].copy()
            matricula = leer_matricula_en_imagen(ocr, plate_crop)
            plate_bbox = (ox + px1, oy + py1, ox + px2, oy + py2)
        else:
            plate_crop = zona.copy()
            matricula = leer_matricula_en_imagen(ocr, plate_crop)
            plate_bbox = (ox, oy, ox + zw, oy + zh)
    else:
        plate_crop = zona.copy()
        matricula = leer_matricula_en_imagen(ocr, plate_crop)
        plate_bbox = (ox, oy, ox + zw, oy + zh)
        plate_conf = 0.0

    out = dict(out_base)
    out.update(
        {
            "matricula": matricula,
            "plate_bbox": plate_bbox,
            "plate_crop": plate_crop,
            "plate_conf": plate_conf,
        }
    )
    return out


def procesar_cajas_tracked(
    imagen, track_result, modelo_placas, ocr,
    conf_min_para_matricula=0.5,
    ids_a_saltar=None,
    escala_x=1.0,
    escala_y=1.0,
):
    """
    A partir del resultado de modelo_coches.track(imagen, persist=True), procesa cada caja
    (coche) con detección de placa y OCR. Devuelve lista de dicts con track_id y plate_conf.

    ids_a_saltar: set de track_ids con matrícula ya confirmada — se incluyen en la salida
                  con sus datos de bbox pero SIN ejecutar OCR (más rápido).
    escala_x/y:   si el frame pasado a YOLO fue redimensionado, escalar las coords de vuelta
                  al tamaño original del frame_detect para que el OCR trabaje sobre pixels reales.
    """
    if imagen is None or imagen.size == 0 or track_result is None or track_result.boxes is None:
        return []
    ids_a_saltar = ids_a_saltar or set()
    salida = []
    for box in track_result.boxes:
        track_id = int(box.id[0]) if getattr(box, "id", None) is not None and len(box.id) > 0 else -1

        # Track confirmado: devolver solo bbox sin OCR
        if track_id in ids_a_saltar:
            cls_id = int(box.cls[0])
            if cls_id not in CLASES_VEHICULOS:
                continue
            # Reescalar bbox al tamaño original si hace falta
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            salida.append({
                "bbox": (x1 * escala_x, y1 * escala_y, x2 * escala_x, y2 * escala_y),
                "clase": track_result.names.get(cls_id, f"clase_{cls_id}"),
                "confianza": float(box.conf[0]),
                "matricula": None,  # ya confirmada, la GUI la tiene
                "plate_bbox": None,
                "plate_crop": None,
                "plate_conf": None,
                "track_id": track_id,
                "confirmado": True,
            })
            continue

        # Reescalar bbox si el frame fue reducido para YOLO
        # Pasamos las coords escaladas directamente a _procesar_una_caja via override
        _escala = (escala_x, escala_y) if (escala_x != 1.0 or escala_y != 1.0) else None

        r = _procesar_una_caja(
            imagen, box, track_result, modelo_placas, ocr, conf_min_para_matricula,
            track_id=track_id, escala=_escala,
        )
        if r is not None:
            salida.append(r)
    return salida


def detectar_vehiculos_y_matriculas_en_frame(
    imagen, modelo_coches, modelo_placas, ocr, conf_min=0.25, conf_min_para_matricula=0.5
):
    """
    Detecta coches con YOLO, dentro de cada coche detecta la matrícula con YOLO-placas,
    y lee el texto con EasyOCR sobre el crop de la placa.
    """
    if imagen is None or imagen.size == 0:
        return []
    device = getattr(modelo_coches, "_use_device", None)
    kwargs = {"conf": conf_min, "verbose": False}
    if device is not None:
        kwargs["device"] = device
    resultados_yolo = modelo_coches(imagen, **kwargs)[0]
    salida = []
    if resultados_yolo.boxes is None:
        return salida
    for box in resultados_yolo.boxes:
        r = _procesar_una_caja(
            imagen, box, resultados_yolo, modelo_placas, ocr, conf_min_para_matricula, track_id=None
        )
        if r is not None:
            r.pop("plate_conf", None)
            salida.append(r)
    return salida


def detectar_vehiculos_y_matriculas(
    ruta_imagen, modelo_coches, modelo_placas, ocr, conf_min=0.25, conf_min_para_matricula=0.5
):
    """
    Detecta vehículos en la imagen y, para cada uno con confianza >= conf_min_para_matricula,
    detecta la placa con YOLO y lee la matrícula con OCR.
    """
    path = Path(ruta_imagen)
    if not path.is_file():
        raise FileNotFoundError(f"No se encuentra la imagen: {ruta_imagen}")
    imagen = cv2.imread(str(path))
    if imagen is None:
        raise ValueError(f"No se pudo leer la imagen: {ruta_imagen}")
    return detectar_vehiculos_y_matriculas_en_frame(
        imagen, modelo_coches, modelo_placas, ocr, conf_min,
        conf_min_para_matricula=conf_min_para_matricula,
    )


def es_video(ruta: str) -> bool:
    """Indica si la ruta corresponde a un vídeo por su extensión."""
    return Path(ruta).suffix.lower() in EXTENSIONES_VIDEO


def procesar_video(
    ruta_video,
    modelo_coches,
    modelo_placas,
    ocr,
    cada_n_frames=10,
    conf_min=0.25,
    conf_min_para_matricula=0.5,
):
    """
    Abre un vídeo y procesa 1 de cada N fotogramas, detectando vehículos y matrículas.
    """
    path = Path(ruta_video)
    if not path.is_file():
        raise FileNotFoundError(f"No se encuentra el vídeo: {ruta_video}")
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"No se pudo abrir el vídeo: {ruta_video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_idx = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break
            if frame_idx % cada_n_frames == 0:
                tiempo_s = frame_idx / fps
                resultados = detectar_vehiculos_y_matriculas_en_frame(
                    frame, modelo_coches, modelo_placas, ocr,
                    conf_min=conf_min,
                    conf_min_para_matricula=conf_min_para_matricula,
                )
                yield frame_idx, tiempo_s, resultados
            frame_idx += 1
    finally:
        cap.release()


def main():
    parser = argparse.ArgumentParser(
        description="Detección de vehículos y matrículas (YOLO + EasyOCR). Acepta imagen o vídeo."
    )
    parser.add_argument(
        "entrada",
        help="Ruta a una imagen (.jpg, .png, ...) o vídeo (.mp4, .avi, ...)",
    )
    parser.add_argument(
        "--cada-n",
        type=int,
        default=10,
        metavar="N",
        help="En vídeo: procesar 1 de cada N fotogramas (por defecto 10)",
    )
    parser.add_argument(
        "--conf-min",
        type=float,
        default=0.25,
        help="Confianza mínima para que YOLO devuelva un vehículo (por defecto 0.25)",
    )
    parser.add_argument(
        "--conf-matricula",
        type=float,
        default=0.5,
        metavar="X",
        help="Solo extraer matrícula si la confianza del vehículo es >= X (coche más completo; por defecto 0.5)",
    )
    args = parser.parse_args()
    ruta = args.entrada

    if not Path(ruta).is_file():
        print(f"Error: no se encuentra el archivo: {ruta}", file=sys.stderr)
        sys.exit(1)

    print("Cargando modelos (YOLO coches, YOLO matrículas, EasyOCR)...")
    modelo_coches, modelo_placas, ocr = cargar_modelos()

    if es_video(ruta):
        print(f"Procesando vídeo (1 de cada {args.cada_n} fotogramas)...")
        for frame_idx, tiempo_s, resultados in procesar_video(
            ruta, modelo_coches, modelo_placas, ocr,
            cada_n_frames=args.cada_n,
            conf_min=args.conf_min,
            conf_min_para_matricula=args.conf_matricula,
        ):
            if resultados:
                print(f"  Frame {frame_idx} (t={tiempo_s:.1f}s): {len(resultados)} vehículo(s)")
                for i, r in enumerate(resultados, 1):
                    mat = r["matricula"] or "(no detectada)"
                    print(f"    {i}. {r['clase']} (conf: {r['confianza']:.2f}) -> Matrícula: {mat}")
    else:
        print("Detectando vehículos y leyendo matrículas...")
        resultados = detectar_vehiculos_y_matriculas(
            ruta, modelo_coches, modelo_placas, ocr,
            conf_min=args.conf_min,
            conf_min_para_matricula=args.conf_matricula,
        )
        print(f"Vehículos detectados: {len(resultados)}")
        for i, r in enumerate(resultados, 1):
            mat = r["matricula"] or "(no detectada)"
            print(f"  {i}. {r['clase']} (conf: {r['confianza']:.2f}) -> Matrícula: {mat}")

    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)

