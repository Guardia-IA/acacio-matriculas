#!/usr/bin/env python3
"""
Test OCR sobre imágenes de matrículas en la carpeta matriculas/.

Para cada imagen:
  - Aplica varios algoritmos de preprocesado (normalize, CLAHE, bilateral, Otsu, sharpen)
  - Ejecuta OCR tradicional (Tesseract), EasyOCR y PaddleOCR
  - Muestra en log el resultado de cada combinación con su % de confianza

Uso:
  python test_ocr.py

Requiere: carpeta matriculas/ con imágenes (jpg, png, etc.)
Tesseract: pip install pytesseract  +  apt install tesseract-ocr  (para OCR tradicional)
PaddleOCR: puede fallar en algunos entornos (error PaddlePaddle); EasyOCR suele funcionar.
"""

import os
import sys
from pathlib import Path

# Evitar comprobación de conectividad de Paddle (acelera arranque)
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

import cv2
import numpy as np

# Extender path para importar del proyecto
PROJECT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_DIR))

MATRICULAS_DIR = PROJECT_DIR / "matriculas"
EXTENSIONES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"}

# Algoritmos de preprocesado (nombre -> función que recibe BGR y devuelve BGR)
def _prep_normalize(img):
    if img is None or img.size == 0:
        return None
    gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gris = cv2.normalize(gris, None, 0, 255, cv2.NORM_MINMAX)
    return cv2.cvtColor(gris, cv2.COLOR_GRAY2BGR)


def _prep_clahe(img):
    if img is None or img.size == 0:
        return None
    gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    gris = clahe.apply(gris)
    return cv2.cvtColor(gris, cv2.COLOR_GRAY2BGR)


def _prep_bilateral(img):
    if img is None or img.size == 0:
        return None
    gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gris = cv2.bilateralFilter(gris, 5, 50, 50)
    gris = cv2.normalize(gris, None, 0, 255, cv2.NORM_MINMAX)
    return cv2.cvtColor(gris, cv2.COLOR_GRAY2BGR)


def _prep_otsu(img):
    if img is None or img.size == 0:
        return None
    gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gris = cv2.bilateralFilter(gris, 5, 50, 50)
    _, gris = cv2.threshold(gris, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return cv2.cvtColor(gris, cv2.COLOR_GRAY2BGR)


def _prep_sharpen(img):
    if img is None or img.size == 0:
        return None
    gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gris = cv2.normalize(gris, None, 0, 255, cv2.NORM_MINMAX)
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    gris = cv2.filter2D(gris, -1, kernel)
    gris = np.clip(gris, 0, 255).astype(np.uint8)
    return cv2.cvtColor(gris, cv2.COLOR_GRAY2BGR)


PREPROCESSORS = {
    "original": lambda img: img,
    "normalize": _prep_normalize,
    "clahe": _prep_clahe,
    "bilateral": _prep_bilateral,
    "otsu": _prep_otsu,
    "sharpen": _prep_sharpen,
}


def ocr_tesseract(img_bgr):
    """Tesseract OCR. Devuelve (texto, confianza)."""
    try:
        import pytesseract
    except ImportError:
        return None, 0.0
    if img_bgr is None or img_bgr.size == 0:
        return None, 0.0
    try:
        data = pytesseract.image_to_data(img_bgr, output_type=pytesseract.Output.DICT, lang="eng")
        texts = []
        confs = []
        for i, text in enumerate(data["text"]):
            t = (text or "").strip()
            if t:
                c = float(data["conf"][i] or 0) / 100.0
                texts.append(t)
                confs.append(c)
        if not texts:
            # Fallback: image_to_string
            txt = pytesseract.image_to_string(img_bgr, lang="eng").strip()
            return txt if txt else None, 0.0
        best = max(range(len(confs)), key=lambda i: (len(texts[i]), confs[i]))
        return " ".join(texts), confs[best]
    except Exception as e:
        print(f"  [Tesseract] Error: {e}", file=sys.stderr)
        return None, 0.0


def ocr_easyocr(reader, img_bgr):
    """EasyOCR. Devuelve (texto, confianza)."""
    if reader is None or img_bgr is None or img_bgr.size == 0:
        return None, 0.0
    try:
        res = reader.readtext(img_bgr)
        if not res:
            return None, 0.0
        # Mejor por longitud y confianza
        best = max(res, key=lambda x: (len((x[1] or "").strip()), x[2]))
        texto = (best[1] or "").strip()
        conf = float(best[2])
        return texto if texto else None, conf
    except Exception as e:
        print(f"  [EasyOCR] Error: {e}", file=sys.stderr)
        return None, 0.0


def ocr_paddle(ocr, img_bgr):
    """PaddleOCR. Devuelve (texto, confianza)."""
    if ocr is None or img_bgr is None or img_bgr.size == 0:
        return None, 0.0
    try:
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        # API antigua: ocr.ocr(); API nueva: ocr.predict() - sin cls
        if hasattr(ocr, "predict"):
            res = ocr.predict(img_rgb)
            # predict puede devolver estructura distinta
            if res is None:
                return None, 0.0
            res = list(res) if not isinstance(res, list) else res
        else:
            res = ocr.ocr(img_rgb, cls=True)
        # Normalizar: res puede ser [[lineas]] o lista de objetos
        if not res:
            return None, 0.0
        lineas = res[0] if isinstance(res, list) and res and isinstance(res[0], list) else res
        if not lineas:
            return None, 0.0
        # Mejor candidato: (línea, conf)
        best_text, best_conf = None, 0.0
        for linea in (lineas if isinstance(lineas, list) else [lineas]):
            if not linea or len(linea) < 2:
                continue
            txt = linea[1][0] if isinstance(linea[1], (list, tuple)) else linea[1]
            conf = linea[1][1] if isinstance(linea[1], (list, tuple)) and len(linea[1]) > 1 else 0.0
            if isinstance(txt, str) and txt.strip() and (best_text is None or conf > best_conf):
                best_text, best_conf = txt.strip(), float(conf)
        return best_text, best_conf
    except Exception as e:
        print(f"  [PaddleOCR] Error: {e}", file=sys.stderr)
        return None, 0.0


def main():
    if not MATRICULAS_DIR.is_dir():
        print(f"No existe la carpeta {MATRICULAS_DIR}", file=sys.stderr)
        print("Crétala y añade imágenes de matrículas (jpg, png, etc.).", file=sys.stderr)
        sys.exit(1)

    imagenes = sorted(p for p in MATRICULAS_DIR.iterdir() if p.suffix.lower() in EXTENSIONES)
    if not imagenes:
        print(f"No hay imágenes en {MATRICULAS_DIR}", file=sys.stderr)
        sys.exit(1)

    print("Cargando motores OCR...", file=sys.stderr)
    reader_easy = None
    try:
        import easyocr
        reader_easy = easyocr.Reader(["en"], gpu=False, verbose=False)
    except Exception as e:
        print(f"EasyOCR no disponible: {e}", file=sys.stderr)

    ocr_paddle_obj = None
    try:
        from paddleocr import PaddleOCR
        ocr_paddle_obj = PaddleOCR(use_textline_orientation=True, lang="en")
    except Exception as e:
        print(f"PaddleOCR no disponible: {e}", file=sys.stderr)

    tesseract_ok = False
    try:
        import pytesseract
        pytesseract.get_tesseract_version()
        tesseract_ok = True
    except Exception:
        print("Tesseract: NO (pip install pytesseract  +  apt install tesseract-ocr)", file=sys.stderr)

    activos = []
    if tesseract_ok:
        activos.append("Tesseract")
    if reader_easy:
        activos.append("EasyOCR")
    if ocr_paddle_obj:
        activos.append("PaddleOCR")
    print(f"Motores activos: {', '.join(activos) or 'ninguno'}", file=sys.stderr)
    print(f"Preprocesados: original, normalize, clahe, bilateral, otsu, sharpen", file=sys.stderr)
    print(f"\nProcesando {len(imagenes)} imagen(es) en {MATRICULAS_DIR}\n", file=sys.stderr)

    for path_img in imagenes:
        print(f"\n{'='*60}")
        print(f"Imagen: {path_img.name}")
        print("=" * 60)

        img = cv2.imread(str(path_img))
        if img is None or img.size == 0:
            print("  [SKIP] No se pudo leer la imagen")
            continue

        for prep_name, prep_fn in PREPROCESSORS.items():
            prep_img = prep_fn(img)
            if prep_img is None:
                continue

            print(f"\n  Preprocesado: {prep_name}")
            print("  " + "-" * 50)

            # Tesseract
            if tesseract_ok:
                txt, conf = ocr_tesseract(prep_img)
                pct = f"{conf*100:.1f}%" if conf else "—"
                print(f"    Tesseract:  {txt or '(sin texto)'}  |  {pct}")

            # EasyOCR
            if reader_easy is not None:
                txt, conf = ocr_easyocr(reader_easy, prep_img)
                pct = f"{conf*100:.1f}%" if conf else "—"
                print(f"    EasyOCR:    {txt or '(sin texto)'}  |  {pct}")

            # PaddleOCR
            if ocr_paddle_obj is not None:
                txt, conf = ocr_paddle(ocr_paddle_obj, prep_img)
                pct = f"{conf*100:.1f}%" if conf else "—"
                print(f"    PaddleOCR:  {txt or '(sin texto)'}  |  {pct}")

    print("\n" + "=" * 60)
    print("Fin del test.")


if __name__ == "__main__":
    main()
