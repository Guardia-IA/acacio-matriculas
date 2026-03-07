"""
Detección de vehículos con YOLO, detección de matrículas con YOLO (modelo específico)
y lectura del texto con OCR (PaddleOCR, EasyOCR o Tesseract).
Flujo: 1) YOLO coches → 2) YOLO matrículas dentro del coche → 3) OCR en el crop de la placa.
"""

import argparse
import re
import sys
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

# OCR backends: import bajo demanda
try:
    import easyocr
except ImportError:
    easyocr = None

try:
    import pytesseract
except ImportError:
    pytesseract = None

def _get_paddleocr():
    try:
        from paddleocr import PaddleOCR
        return PaddleOCR
    except ImportError:
        return None


def _get_platerec():
    try:
        from platerec import Platerec
        return Platerec
    except ImportError:
        return None


def _get_fastalpr():
    """Devuelve la clase ALPR de fast_alpr si está instalado."""
    try:
        from fast_alpr import ALPR
        return ALPR
    except ImportError:
        return None


# Dispositivo para inferencia: se establece en cargar_modelos()
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

# Configuración Tesseract para matrículas: una línea, solo letras mayúsculas y dígitos
TESSERACT_CONFIG_MATRICULA = (
    "--psm 7 "  # una línea de texto
    "--oem 3 "   # LSTM + legacy
    "-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
)

# Prefijo de país que platerec añade (ej. [ue], [br]); lo quitamos para comparar matrícula
RE_PLATEREC_PREFIX = re.compile(r"^\[[a-z]+\]\s*", re.I)


def _is_fastalpr(ocr):
    """True si ocr es una instancia de fast_alpr.ALPR (detección+OCR en uno)."""
    if ocr is None:
        return False
    return type(ocr).__name__ == "ALPR"


def _extraer_matricula_fastalpr(alpr, zona_bgr, ox, oy):
    """
    Ejecuta fast-alpr sobre la zona (crop del coche) y devuelve la mejor matrícula.
    Returns: (matricula, plate_bbox, plate_crop, plate_conf) o (None, None, None, 0.0).
    """
    if zona_bgr is None or zona_bgr.size == 0:
        return None, None, None, 0.0
    try:
        results = alpr.predict(zona_bgr)
    except Exception:
        return None, None, None, 0.0
    if not results:
        return None, None, None, 0.0
    zh, zw = zona_bgr.shape[:2]
    best = None
    best_score = -1.0
    for r in results:
        if r.ocr is None or not (getattr(r.ocr, "text", None) or "").strip():
            continue
        text = (r.ocr.text or "").strip().upper()
        text = limpiar_texto_matricula(text)
        if not parece_matricula(text):
            continue
        conf = r.ocr.confidence
        if isinstance(conf, (list, tuple)):
            conf = float(conf[0]) if conf else 0.5
        else:
            conf = float(conf)
        det = r.detection
        b = getattr(det, "bounding_box", None)
        if b is None:
            continue
        x1 = max(0, getattr(b, "x1", 0))
        y1 = max(0, getattr(b, "y1", 0))
        x2 = min(zw, getattr(b, "x2", zw))
        y2 = min(zh, getattr(b, "y2", zh))
        if x2 <= x1 or y2 <= y1:
            continue
        score = conf * 40.0 + len(text) * 18.0
        if score > best_score:
            best_score = score
            best = (text, (ox + x1, oy + y1, ox + x2, oy + y2), zona_bgr[y1:y2, x1:x2].copy(), conf)
    if best is None:
        return None, None, None, 0.0
    return best[0], best[1], best[2], best[3]


class PlateRecOCR:
    """
    Wrapper de platerec (plate-recognition) con interfaz .readtext(img) -> [(box, text, conf), ...].
    Lee solo el crop de placa; quita el prefijo de país [ue]/[br] del resultado.
    """

    def __init__(self, providers=None):
        Platerec = _get_platerec()
        if Platerec is None:
            raise RuntimeError("platerec no instalado. pip install \"platerec[cpu]\"")
        self._pr = Platerec(providers=providers or ["CPUExecutionProvider"])

    def readtext(self, img):
        """Ejecuta platerec sobre el crop BGR y devuelve [(box, texto, confianza), ...]."""
        if img is None or img.size == 0:
            return []
        try:
            from PIL import Image
            pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            pred = self._pr.read(pil)
            if not pred or "word" not in pred:
                return []
            word = (pred.get("word") or "").strip()
            word = RE_PLATEREC_PREFIX.sub("", word).strip()
            conf = float(pred.get("confidence", 0.5))
            if not word:
                return []
            # Bbox dummy (platerec no devuelve caja para read())
            h, w = img.shape[:2]
            box = [[0, 0], [w, 0], [w, h], [0, h]]
            return [(box, word, conf)]
        except Exception:
            return []


class PaddleOCRWrapper:
    """
    Wrapper de PaddleOCR con interfaz compatible con EasyOCR: .readtext(img) -> [(box, text, conf), ...].
    """

    def __init__(self, use_angle_cls=True, use_gpu=False, lang="en", enable_mkldnn=False):
        PaddleOCR = _get_paddleocr()
        if PaddleOCR is None:
            raise RuntimeError("paddleocr no instalado. pip install paddlepaddle paddleocr")
        # PaddleOCR 3.x: device=, use_textline_orientation=, enable_mkldnn=False evita error OneDNN en algunos CPU
        try:
            self._ocr = PaddleOCR(
                lang=lang,
                use_textline_orientation=use_angle_cls,
                device="cpu" if not use_gpu else "gpu",
                enable_mkldnn=enable_mkldnn,
            )
        except (TypeError, ValueError):
            self._ocr = PaddleOCR(use_angle_cls=use_angle_cls, lang=lang, use_gpu=use_gpu)

    def readtext(self, img):
        """
        Ejecuta PaddleOCR y devuelve lista en formato EasyOCR: [(bbox, texto, confianza), ...].
        Soporta tanto la API antigua .ocr() como la nueva .predict() (PaddleOCR 3.x).
        """
        if img is None or img.size == 0:
            return []
        try:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if hasattr(self._ocr, "predict"):
                res = self._ocr.predict(img_rgb)
            else:
                res = self._ocr.ocr(img_rgb, cls=True)
            if not res:
                return []
            out = []
            # PaddleOCR 3.x: predict() devuelve lista de dicts con rec_texts, rec_scores, rec_polys
            if isinstance(res, list) and res and isinstance(res[0], dict):
                for page in res:
                    texts = page.get("rec_texts") or []
                    scores = page.get("rec_scores") or []
                    polys = page.get("rec_polys") or []
                    for i, texto in enumerate(texts):
                        if not isinstance(texto, str) or not texto.strip():
                            continue
                        conf = float(scores[i]) if i < len(scores) else 0.5
                        if conf > 1:
                            conf = conf / 100.0
                        box = polys[i].tolist() if i < len(polys) and hasattr(polys[i], "tolist") else [[0, 0], [0, 0], [0, 0], [0, 0]]
                        if isinstance(box, list) and box and isinstance(box[0], (list, tuple)):
                            pass  # ya es lista de puntos
                        else:
                            box = [[0, 0], [0, 0], [0, 0], [0, 0]]
                        out.append((box, texto.strip(), conf))
                return out
            # API antigua: [[[box, (text, conf)], ...]]
            if isinstance(res[0], list):
                lineas = res[0]
            else:
                lineas = res if isinstance(res, list) else []
            for linea in lineas:
                if not linea or len(linea) < 2:
                    continue
                box = linea[0]
                texto_conf = linea[1]
                if isinstance(texto_conf, (list, tuple)):
                    texto = texto_conf[0] if len(texto_conf) > 0 else ""
                    conf = float(texto_conf[1]) if len(texto_conf) > 1 else 0.5
                else:
                    texto, conf = str(texto_conf), 0.5
                if not isinstance(texto, str) or not texto.strip():
                    continue
                if conf > 1:
                    conf = conf / 100.0
                out.append((box, texto.strip(), conf))
            return out
        except Exception:
            return []


class TesseractOCR:
    """
    Wrapper de Tesseract con interfaz compatible con EasyOCR: .readtext(img) -> [(box, text, conf), ...].
    Tuneado para matrículas (solo A-Z y 0-9, PSM 7).
    """

    def __init__(self, config=None):
        if pytesseract is None:
            raise RuntimeError("pytesseract no instalado. pip install pytesseract y apt install tesseract-ocr")
        self.config = config or TESSERACT_CONFIG_MATRICULA

    def readtext(self, img):
        """
        Ejecuta Tesseract sobre la imagen y devuelve lista en formato EasyOCR:
        [(bbox, texto, confianza), ...] con confianza en 0.0-1.0.
        """
        if img is None or img.size == 0:
            return []
        try:
            data = pytesseract.image_to_data(
                img,
                config=self.config,
                output_type=pytesseract.Output.DICT,
                lang="eng",
            )
        except Exception:
            return []
        n = len(data["text"])
        resultados = []
        for i in range(n):
            texto = (data["text"][i] or "").strip()
            if not texto:
                continue
            conf = float(data["conf"][i])
            # Tesseract puede devolver conf=0 o -1 aunque haya texto; normalizar a [0, 1]
            if conf < 0:
                conf = 0.0
            conf_pct = conf / 100.0 if conf > 0 else 0.5
            left = data["left"][i]
            top = data["top"][i]
            w = data["width"][i]
            h = data["height"][i]
            # Bbox como 4 puntos [[x1,y1],[x2,y2],[x3,y3],[x4,y4]] (formato EasyOCR)
            box = [
                [left, top],
                [left + w, top],
                [left + w, top + h],
                [left, top + h],
            ]
            resultados.append((box, texto, conf_pct))
        return resultados


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


def cargar_modelos(yolo_coches: str = "yolo11n.pt", yolo_placas=None, ocr_backend: str = "easyocr"):
    """
    Carga: YOLO coches, YOLO matrículas (modelo específico) y OCR.

    ocr_backend: "easyocr" (por defecto), "platerec", "paddleocr" o "tesseract".
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

        # --- OCR: EasyOCR (por defecto), platerec, PaddleOCR, Tesseract o fastalpr ---
        backend = (ocr_backend or "easyocr").lower()
        if backend == "easyocr":
            if easyocr is None:
                raise RuntimeError("easyocr no instalado. pip install easyocr")
            print("  Cargando EasyOCR (CPU)...")
            ocr = easyocr.Reader(["en"], gpu=False, verbose=False)
            print("  EasyOCR listo.")
        elif backend == "platerec":
            print("  Cargando platerec (plate-recognition ONNX)...")
            ocr = PlateRecOCR()
            print("  platerec listo.")
        elif backend == "tesseract":
            print("  Usando Tesseract OCR (tuneado para matrículas)...")
            ocr = TesseractOCR()
            print("  Tesseract listo.")
        elif backend == "paddleocr":
            print("  Cargando PaddleOCR (CPU, sin MKLDNN para evitar error OneDNN)...")
            ocr = PaddleOCRWrapper(use_angle_cls=True, use_gpu=False, lang="en", enable_mkldnn=False)
            print("  PaddleOCR listo.")
        elif backend == "fastalpr":
            ALPR = _get_fastalpr()
            if ALPR is None:
                raise RuntimeError("fast-alpr no instalado. pip install fast-alpr[onnx]")
            print("  Cargando fast-alpr (detección + OCR ONNX)...")
            ocr = ALPR()
            print("  fast-alpr listo.")
        else:
            if easyocr is None:
                raise RuntimeError("easyocr no instalado. pip install easyocr")
            print("  Cargando EasyOCR (CPU)...")
            ocr = easyocr.Reader(["en"], gpu=False, verbose=False)
            print("  EasyOCR listo.")
        return modelo_coches, modelo_placas, ocr
    finally:
        sys.argv[:] = argv_orig


# EasyOCR es el OCR principal (sin dependencia de cuDNN del sistema)


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


# Altura objetivo del crop de placa para OCR: siempre escalamos a al menos esto ("zoom")
# Para vídeo 4K el crop puede ser pequeño; upscalar mejora claridad para EasyOCR
ALTURA_MIN_PLACA_OCR = 48
ALTURA_PLACA_OBJETIVO = 160  # Zoom: más altura para placas lejanas (mejor distinción 6/B, 8/B)
# Tamaño fijo del crop antes de EasyOCR (ancho x alto) — más resolución ayuda cuando el coche está lejos
ANCHO_CROP_OCR = 320
ALTO_CROP_OCR = 64
# Margen al recortar la zona de texto dentro del crop (evitar cortar caracteres)
MARGEN_RECORTE_PLACA = 0.08  # 8% del ancho/alto mínimo
# Margen alrededor del bbox de la placa (YOLO): mínimo 4 px para no cortar bordes
MARGEN_BBOX_PLACA = 0  # 0 = solo margen mínimo de 4 px; subir (ej. 0.08–0.12) si recorta caracteres


def recortar_y_centrar_placa(imagen):
    """
    Recorta el crop de placa para quedarnos solo con la zona que tiene contenido (texto).
    Usa contornos o umbral para encontrar la región no vacía y aplica un margen.
    Si no se encuentra región válida, devuelve la imagen original.
    """
    if imagen is None or imagen.size == 0:
        return imagen
    h, w = imagen.shape[:2]
    if h < 6 or w < 20:
        return imagen
    gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY) if len(imagen.shape) == 3 else imagen
    try:
        binaria = cv2.adaptiveThreshold(
            gris, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
        )
    except Exception:
        _, binaria = cv2.threshold(gris, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    binaria = cv2.morphologyEx(binaria, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(
        binaria, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return imagen
    min_area = (w * h) * 0.02
    rects = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area:
            continue
        x, y, rw, rh = cv2.boundingRect(c)
        if rw < 5 or rh < 3:
            continue
        rects.append((x, y, x + rw, y + rh))
    if not rects:
        return imagen
    x1 = min(r[0] for r in rects)
    y1 = min(r[1] for r in rects)
    x2 = max(r[2] for r in rects)
    y2 = max(r[3] for r in rects)
    m = max(1, int(min(x2 - x1, y2 - y1) * MARGEN_RECORTE_PLACA))
    x1 = max(0, x1 - m)
    y1 = max(0, y1 - m)
    x2 = min(w, x2 + m)
    y2 = min(h, y2 + m)
    if x2 <= x1 or y2 <= y1:
        return imagen
    return imagen[y1:y2, x1:x2].copy()


def trim_bordes_placa(imagen, umbral_std_min=5):
    """
    Recorta filas/columnas de los bordes con poca variación (fondo uniforme).
    Centra mejor la placa cuando el bbox de YOLO trae mucho margen.
    """
    if imagen is None or imagen.size == 0:
        return imagen
    gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY) if len(imagen.shape) == 3 else imagen
    h, w = gris.shape[:2]
    if h < 4 or w < 10:
        return imagen
    std_f = np.std(gris, axis=1)
    std_c = np.std(gris, axis=0)
    y_ini, y_fin = 0, h
    for i in range(h):
        if std_f[i] >= umbral_std_min:
            y_ini = i
            break
    for i in range(h - 1, -1, -1):
        if std_f[i] >= umbral_std_min:
            y_fin = i + 1
            break
    x_ini, x_fin = 0, w
    for j in range(w):
        if std_c[j] >= umbral_std_min:
            x_ini = j
            break
    for j in range(w - 1, -1, -1):
        if std_c[j] >= umbral_std_min:
            x_fin = j + 1
            break
    if x_fin <= x_ini or y_fin <= y_ini:
        return imagen
    nw = x_fin - x_ini
    nh = y_fin - y_ini
    if nw < 10 or nh < 4:
        return imagen
    return imagen[y_ini:y_fin, x_ini:x_fin].copy()


def resize_placa_250x50(imagen):
    """
    Redimensiona el crop de placa a ANCHO_CROP_OCR x ALTO_CROP_OCR antes del OCR.
    Usa INTER_LANCZOS4 para mejor nitidez al escalar.
    """
    if imagen is None or imagen.size == 0:
        return imagen
    return cv2.resize(
        imagen, (ANCHO_CROP_OCR, ALTO_CROP_OCR), interpolation=cv2.INTER_LANCZOS4
    )


def preparar_crop_placa_para_ocr(imagen):
    """
    Recorta y centra la zona de texto de la placa y quita bordes vacíos.
    Devuelve el crop listo para resize y variantes de preprocesado.
    Si el recorte falla o deja algo demasiado pequeño, devuelve la imagen original.
    En crops muy pequeños (placa lejana) solo hace trim para no cortar caracteres.
    """
    if imagen is None or imagen.size == 0:
        return imagen
    h, w = imagen.shape[:2]
    # En placas lejanas (crop pequeño) no usar contornos: riesgo de cortar 6/B, 8/B
    if h >= 22 and w >= 60:
        recortada = recortar_y_centrar_placa(imagen)
        if recortada.shape[1] < 15 or recortada.shape[0] < 5:
            recortada = imagen
    else:
        recortada = imagen
    recortada = trim_bordes_placa(recortada)
    rh, rw = recortada.shape[:2]
    if rw < 10 or rh < 4:
        return imagen
    return recortada


def _preprocesar_variante_clahe(imagen):
    """CLAHE: mejora contraste local. Muy efectivo para placas con mala iluminación."""
    if imagen is None or imagen.size == 0:
        return None
    h, w = imagen.shape[:2]
    gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    gris = clahe.apply(gris)
    return cv2.cvtColor(gris, cv2.COLOR_GRAY2BGR)


def _preprocesar_variante_bilateral(imagen):
    """Bilateral: reduce ruido manteniendo bordes (números más nítidos)."""
    if imagen is None or imagen.size == 0:
        return None
    gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    gris = cv2.bilateralFilter(gris, 5, 50, 50)
    gris = cv2.normalize(gris, None, 0, 255, cv2.NORM_MINMAX)
    return cv2.cvtColor(gris, cv2.COLOR_GRAY2BGR)


def _preprocesar_variante_otsu(imagen):
    """Otsu: binarización adaptativa. Útil cuando hay buen contraste fondo/texto."""
    if imagen is None or imagen.size == 0:
        return None
    gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    gris = cv2.bilateralFilter(gris, 5, 50, 50)
    _, gris = cv2.threshold(gris, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return cv2.cvtColor(gris, cv2.COLOR_GRAY2BGR)


def _preprocesar_variante_sharpen(imagen):
    """Sharpen: realza bordes para caracteres más legibles."""
    if imagen is None or imagen.size == 0:
        return None
    gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    gris = cv2.normalize(gris, None, 0, 255, cv2.NORM_MINMAX)
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])  # sharpen
    gris = cv2.filter2D(gris, -1, kernel)
    gris = np.clip(gris, 0, 255).astype(np.uint8)
    return cv2.cvtColor(gris, cv2.COLOR_GRAY2BGR)


def _preprocesar_variante_unsharp(imagen, sigma=1.0, strength=1.4):
    """Unsharp mask: más nitidez para mejorar lectura OCR."""
    if imagen is None or imagen.size == 0:
        return None
    blurred = cv2.GaussianBlur(imagen, (0, 0), sigma)
    sharp = cv2.addWeighted(imagen, strength, blurred, -(strength - 1), 0)
    return np.clip(sharp, 0, 255).astype(np.uint8)


def _preprocesar_variante_morfologia(imagen):
    """Apertura morfológica suave: elimina ruido fino manteniendo caracteres."""
    if imagen is None or imagen.size == 0:
        return None
    gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    gris = cv2.normalize(gris, None, 0, 255, cv2.NORM_MINMAX)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    gris = cv2.morphologyEx(gris, cv2.MORPH_OPEN, kernel)
    return cv2.cvtColor(gris, cv2.COLOR_GRAY2BGR)


def zoom_placa_para_ocr(imagen):
    """
    Hace zoom (upscale) del crop de placa para mejorar resolución antes del OCR.
    Usa INTER_LANCZOS4 para mejor nitidez al ampliar.
    """
    if imagen is None or imagen.size == 0:
        return imagen
    h, w = imagen.shape[:2]
    if h < ALTURA_PLACA_OBJETIVO and h > 0:
        escala = ALTURA_PLACA_OBJETIVO / h
        nw = max(1, int(w * escala))
        nh = ALTURA_PLACA_OBJETIVO
        return cv2.resize(imagen, (nw, nh), interpolation=cv2.INTER_LANCZOS4)
    return imagen


def preprocesar_placa_para_ocr(imagen):
    """
    Zoom + mejora de contraste del crop de placa para OCR.
    """
    if imagen is None or imagen.size == 0:
        return imagen
    imagen = zoom_placa_para_ocr(imagen)
    h, w = imagen.shape[:2]
    if h < ALTURA_MIN_PLACA_OCR and h > 0:
        escala = ALTURA_MIN_PLACA_OCR / h
        nw = max(1, int(w * escala))
        nh = ALTURA_MIN_PLACA_OCR
        imagen = cv2.resize(imagen, (nw, nh), interpolation=cv2.INTER_LANCZOS4)
    gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    gris = cv2.normalize(gris, None, 0, 255, cv2.NORM_MINMAX)
    return cv2.cvtColor(gris, cv2.COLOR_GRAY2BGR)


def preprocesar_variantes_placa(imagen):
    """
    Genera varias variantes preprocesadas de la placa para probar con OCR.
    Cada variante puede funcionar mejor según iluminación, ángulo, ruido.
    Devuelve lista de imágenes BGR.
    """
    if imagen is None or imagen.size == 0:
        return []
    h, w = imagen.shape[:2]
    if h < ALTURA_MIN_PLACA_OCR and h > 0:
        escala = ALTURA_MIN_PLACA_OCR / h
        nw = max(1, int(w * escala))
        nh = ALTURA_MIN_PLACA_OCR
        imagen = cv2.resize(imagen, (nw, nh), interpolation=cv2.INTER_LANCZOS4)
    variantes = []
    try:
        variantes.append(_preprocesar_variante_clahe(imagen))
    except Exception:
        pass
    try:
        variantes.append(_preprocesar_variante_bilateral(imagen))
    except Exception:
        pass
    try:
        variantes.append(_preprocesar_variante_otsu(imagen))
    except Exception:
        pass
    try:
        variantes.append(_preprocesar_variante_sharpen(imagen))
    except Exception:
        pass
    # Base: normalize (siempre incluida)
    gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    gris = cv2.normalize(gris, None, 0, 255, cv2.NORM_MINMAX)
    variantes.insert(0, cv2.cvtColor(gris, cv2.COLOR_GRAY2BGR))
    return [v for v in variantes if v is not None and v.size > 0]


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


def _extraer_matricula_con_confianza_easyocr(resultados_easyocr):
    """De EasyOCR devuelve (mejor_matricula, confianza) o (None, 0.0)."""
    if not resultados_easyocr:
        return None, 0.0
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
        return None, 0.0
    candidatos.sort(key=lambda x: (-len(x[0]), -x[1]))
    return candidatos[0][0], float(candidatos[0][1])


def _extraer_matricula_de_easyocr(resultados_easyocr):
    """De la lista [(bbox, texto, conf), ...] de EasyOCR devuelve el mejor candidato a matrícula."""
    txt, _ = _extraer_matricula_con_confianza_easyocr(resultados_easyocr)
    return txt


def leer_matricula_con_confianza_cualquier_backend(ocr, imagen):
    """
    Devuelve (matricula, confianza) para cualquier backend (EasyOCR, platerec, ALPR, etc.).
    Con fast-alpr (ALPR) usa predict() en lugar de readtext().
    """
    if imagen is None or imagen.size == 0:
        return None, 0.0
    if _is_fastalpr(ocr):
        try:
            results = ocr.predict(imagen)
        except Exception:
            return None, 0.0
        if not results:
            return None, 0.0
        best_text = None
        best_conf = 0.0
        for r in results:
            if r.ocr is None or not (getattr(r.ocr, "text", None) or "").strip():
                continue
            text = limpiar_texto_matricula((r.ocr.text or "").strip())
            if not parece_matricula(text):
                continue
            conf = r.ocr.confidence
            conf = float(conf[0]) if isinstance(conf, (list, tuple)) and conf else float(conf)
            if conf > best_conf or (best_text is None and text):
                best_text, best_conf = text, conf
        return best_text or None, best_conf
    return leer_matricula_con_confianza(ocr, imagen)


def leer_matricula_en_imagen(ocr, imagen):
    """
    Lee la matrícula en el crop usando EasyOCR.
    Primero recorta/centra la placa, luego 250x50, limpia y afila (zoom, CLAHE, bilateral, sharpen, unsharp) y prueba varias variantes.
    """
    if imagen is None or imagen.size == 0:
        return None

    imagen = preparar_crop_placa_para_ocr(imagen)
    imagen = resize_placa_250x50(imagen)
    img_zoom = zoom_placa_para_ocr(imagen)
    variantes = [
        preprocesar_placa_para_ocr(imagen),
        img_zoom,
        _preprocesar_variante_unsharp(imagen),
        _preprocesar_variante_unsharp(img_zoom),
    ]
    try:
        variantes.insert(2, _preprocesar_variante_clahe(img_zoom))
    except Exception:
        pass
    try:
        variantes.append(_preprocesar_variante_bilateral(img_zoom))
    except Exception:
        pass
    try:
        variantes.append(_preprocesar_variante_sharpen(img_zoom))
    except Exception:
        pass
    try:
        variantes.append(_preprocesar_variante_otsu(img_zoom))
    except Exception:
        pass
    try:
        variantes.append(_preprocesar_variante_morfologia(img_zoom))
    except Exception:
        pass
    variantes = [v for v in variantes if v is not None and v.size > 0]
    if not variantes:
        variantes = [imagen]

    for img in variantes:
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


def leer_matricula_con_confianza(ocr, imagen):
    """
    Lee la matrícula en el crop y devuelve (texto, confianza) o (None, 0.0).
    Primero recorta/centra la placa, mismo preprocesado que leer_matricula_en_imagen; devuelve la mejor lectura por confianza.
    """
    if imagen is None or imagen.size == 0:
        return None, 0.0

    imagen = preparar_crop_placa_para_ocr(imagen)
    imagen = resize_placa_250x50(imagen)
    img_zoom = zoom_placa_para_ocr(imagen)
    variantes = [
        preprocesar_placa_para_ocr(imagen),
        img_zoom,
        _preprocesar_variante_unsharp(imagen),
        _preprocesar_variante_unsharp(img_zoom),
    ]
    try:
        variantes.insert(2, _preprocesar_variante_clahe(img_zoom))
    except Exception:
        pass
    try:
        variantes.append(_preprocesar_variante_bilateral(img_zoom))
    except Exception:
        pass
    try:
        variantes.append(_preprocesar_variante_sharpen(img_zoom))
    except Exception:
        pass
    try:
        variantes.append(_preprocesar_variante_otsu(img_zoom))
    except Exception:
        pass
    try:
        variantes.append(_preprocesar_variante_morfologia(img_zoom))
    except Exception:
        pass
    variantes = [v for v in variantes if v is not None and v.size > 0]
    if not variantes:
        variantes = [imagen]

    mejor_texto, mejor_conf = None, 0.0
    for img in variantes:
        if img is None or img.size == 0:
            continue
        try:
            res = ocr.readtext(img)
            texto, conf = _extraer_matricula_con_confianza_easyocr(res)
            if texto and (mejor_texto is None or conf > mejor_conf or (conf >= mejor_conf and len(texto) > len(mejor_texto or ""))):
                mejor_texto, mejor_conf = texto, conf
        except Exception:
            pass
    return mejor_texto, mejor_conf


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
        "plate_crop_processed": None,
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

    if _is_fastalpr(ocr):
        # fast-alpr: detección + OCR en uno sobre la zona del coche (no usa YOLO placas)
        matricula, plate_bbox, plate_crop, plate_conf = _extraer_matricula_fastalpr(ocr, zona, ox, oy)
        if plate_crop is None:
            plate_crop = zona.copy()
            plate_conf = 0.0
        if plate_bbox is None:
            plate_bbox = (ox, oy, ox + zw, oy + zh)
    else:
        plate_bbox_crop, plate_conf = detectar_placa_en_crop(modelo_placas, zona, conf_min=0.25)
        if plate_bbox_crop is not None:
            px1, py1, px2, py2 = [int(v) for v in plate_bbox_crop]
            # Margen alrededor del bbox para no cortar el primer/último carácter (mejor OCR en 6/B, 8/B)
            pw = max(1, px2 - px1)
            ph = max(1, py2 - py1)
            m = max(4, int(pw * MARGEN_BBOX_PLACA), int(ph * MARGEN_BBOX_PLACA))
            px1 = max(0, px1 - m)
            py1 = max(0, py1 - m)
            px2 = min(zw, px2 + m)
            py2 = min(zh, py2 + m)
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

    # Versión procesada (recorte + limpieza + resize): para mostrar en interfaz y subir
    plate_crop_processed = None
    if plate_crop is not None and plate_crop.size > 0:
        try:
            plate_crop_processed = resize_placa_250x50(preparar_crop_placa_para_ocr(plate_crop))
            if plate_crop_processed is not None and plate_crop_processed.size > 0:
                pass  # ya asignado
            else:
                plate_crop_processed = None
        except Exception:
            plate_crop_processed = None

    out = dict(out_base)
    out.update(
        {
            "matricula": matricula,
            "plate_bbox": plate_bbox,
            "plate_crop": plate_crop,
            "plate_crop_processed": plate_crop_processed,
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

