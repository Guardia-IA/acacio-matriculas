"""
Prueba diferentes métodos de preprocesado/sharpening sobre un crop de matrícula
y compara resultados de EasyOCR y Tesseract.
Uso: python test_scale.py [ruta_imagen]
"""

import sys
from pathlib import Path

import cv2
import numpy as np

# Ruta por defecto: imagen de prueba (assets de Cursor)
IMAGEN_DEFECTO = Path.home() / ".cursor" / "projects" / "home-debian-dev-Proyectos-AcacioMatriculas-acacio-matriculas" / "assets" / "image-42ccdc8d-d903-4e22-86a1-72fe620cb825.png"

# Fallback si no existe: matriculas/
IMAGEN_FALLBACK = Path(__file__).resolve().parent / "matriculas" / "matricula1.jpg"


def _resize_min_height(img, min_h=120):
    """Escala la imagen para tener al menos min_h px de altura."""
    if img is None or img.size == 0:
        return img
    h, w = img.shape[:2]
    if h < min_h and h > 0:
        escala = min_h / h
        nw, nh = max(1, int(w * escala)), min_h
        return cv2.resize(img, (nw, nh), interpolation=cv2.INTER_CUBIC)
    return img


def metodo_original(img):
    """Sin procesar."""
    return img.copy()


def metodo_unsharp_mask(img, sigma=1.0, strength=1.5):
    """Unsharp mask: aumenta nitidez restando una versión difuminada."""
    blurred = cv2.GaussianBlur(img, (0, 0), sigma)
    sharp = cv2.addWeighted(img, strength, blurred, -(strength - 1), 0)
    return np.clip(sharp, 0, 255).astype(np.uint8)


def metodo_kernel_sharpen(img):
    """Sharpen con kernel 3x3."""
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    return cv2.filter2D(img, -1, kernel)


def metodo_pil_sharpness(img, factor=2.0):
    """PIL ImageEnhance.Sharpness."""
    from PIL import Image, ImageEnhance
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    enh = ImageEnhance.Sharpness(pil)
    sharp_pil = enh.enhance(factor)
    return cv2.cvtColor(np.array(sharp_pil), cv2.COLOR_RGB2BGR)


def metodo_lanczos_upscale(img, factor=2.0):
    """Upscale con INTER_LANCZOS4 (mejor calidad que CUBIC)."""
    h, w = img.shape[:2]
    nw, nh = int(w * factor), int(h * factor)
    return cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LANCZOS4)


def metodo_clahe(img):
    """CLAHE: contraste adaptativo local."""
    gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    gris = clahe.apply(gris)
    return cv2.cvtColor(gris, cv2.COLOR_GRAY2BGR)


def metodo_bilateral(img):
    """Bilateral: reduce ruido manteniendo bordes."""
    gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gris = cv2.bilateralFilter(gris, 5, 50, 50)
    gris = cv2.normalize(gris, None, 0, 255, cv2.NORM_MINMAX)
    return cv2.cvtColor(gris, cv2.COLOR_GRAY2BGR)


def metodo_otsu(img):
    """Otsu: binarización adaptativa."""
    gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gris = cv2.bilateralFilter(gris, 5, 50, 50)
    _, gris = cv2.threshold(gris, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return cv2.cvtColor(gris, cv2.COLOR_GRAY2BGR)


def metodo_normalize(img):
    """Normalize + escala mínima."""
    img = _resize_min_height(img, 120)
    gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gris = cv2.normalize(gris, None, 0, 255, cv2.NORM_MINMAX)
    return cv2.cvtColor(gris, cv2.COLOR_GRAY2BGR)


def metodo_combo_upscale_unsharp(img):
    """Combinación: upscale 2x + unsharp mask."""
    up = metodo_lanczos_upscale(img, 2.0)
    return metodo_unsharp_mask(up, sigma=1.0, strength=1.3)


def metodo_combo_upscale_sharpen(img):
    """Combinación: upscale 2x + kernel sharpen."""
    up = metodo_lanczos_upscale(img, 2.0)
    return metodo_kernel_sharpen(up)


def metodo_combo_clahe_unsharp(img):
    """Combinación: CLAHE + unsharp."""
    c = metodo_clahe(img)
    return metodo_unsharp_mask(c, sigma=0.8, strength=1.2)


# Lista de todos los métodos a probar
METODOS = [
    ("original", metodo_original),
    ("normalize+zoom120", metodo_normalize),
    ("unsharp_mask", metodo_unsharp_mask),
    ("kernel_sharpen", metodo_kernel_sharpen),
    ("pil_sharpness", metodo_pil_sharpness),
    ("lanczos_upscale_2x", metodo_lanczos_upscale),
    ("clahe", metodo_clahe),
    ("bilateral", metodo_bilateral),
    ("otsu", metodo_otsu),
    ("combo: upscale+unsharp", metodo_combo_upscale_unsharp),
    ("combo: upscale+sharpen", metodo_combo_upscale_sharpen),
    ("combo: clahe+unsharp", metodo_combo_clahe_unsharp),
]


def run_easyocr(img, reader):
    """Ejecuta EasyOCR y devuelve (texto, confianza) del mejor candidato."""
    try:
        results = reader.readtext(img)
        if not results:
            return None, 0.0
        # Mejor por confianza
        best = max(results, key=lambda r: r[2])
        return best[1].strip(), float(best[2])
    except Exception as e:
        return f"ERR:{e}", 0.0


def run_tesseract(img):
    """Ejecuta Tesseract y devuelve (texto, confianza)."""
    try:
        import pytesseract
        txt = pytesseract.image_to_string(img).strip()
        data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
        confs = [data["conf"][i] for i in range(len(data["text"])) if data["text"][i].strip()]
        conf = max(confs) if confs else 0
        return txt if txt else None, float(conf)
    except Exception as e:
        return f"ERR:{e}", 0.0


def main():
    ruta = sys.argv[1] if len(sys.argv) > 1 else None
    if ruta is None:
        for p in [IMAGEN_DEFECTO, IMAGEN_FALLBACK]:
            if p.exists():
                ruta = str(p)
                break
    if ruta is None:
        ruta = str(IMAGEN_FALLBACK)
        print(f"No encontrada imagen por defecto. Usando: {ruta}")

    img = cv2.imread(ruta)
    if img is None or img.size == 0:
        print(f"Error: no se pudo cargar {ruta}")
        sys.exit(1)

    print(f"Imagen: {ruta}")
    print(f"Tamaño: {img.shape[1]}x{img.shape[0]}")
    print("-" * 90)

    # Cargar EasyOCR una sola vez (evita crear 12 lectores)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        print("Cargando EasyOCR (puede tardar unos segundos)...")
        import easyocr
        reader = easyocr.Reader(["en"], gpu=False, verbose=False)
    print("Listo.")
    print()

    resultados = []

    for nombre, metodo in METODOS:
        try:
            proc = metodo(img)
            if proc is None or proc.size == 0:
                eocr_txt, eocr_conf = "N/A", 0.0
                tess_txt, tess_conf = "N/A", 0.0
            else:
                eocr_txt, eocr_conf = run_easyocr(proc, reader)
                tess_txt, tess_conf = run_tesseract(proc)
        except Exception as e:
            eocr_txt, eocr_conf = f"ERR:{e}", 0.0
            tess_txt, tess_conf = f"ERR:{e}", 0.0

        eocr_str = f"{eocr_txt} ({eocr_conf:.3f})" if eocr_txt else "—"
        tess_str = f"{tess_txt} ({tess_conf:.1f})" if tess_txt else "—"
        resultados.append((nombre, eocr_str, tess_str, eocr_conf, tess_conf))

    # Tabla
    print(f"{'Método':<30} | {'EasyOCR':<35} | {'Tesseract':<35}")
    print("-" * 105)
    for nombre, eocr, tess, _, _ in resultados:
        print(f"{nombre:<30} | {eocr:<35} | {tess:<35}")

    # Mejor por EasyOCR
    ok = [r for r in resultados if r[3] > 0]
    if ok:
        best_eocr = max(ok, key=lambda r: r[3])
        print()
        print(f"Mejor EasyOCR: {best_eocr[0]} -> {best_eocr[1]}")
    ok_t = [r for r in resultados if r[4] > 0]
    if ok_t:
        best_tess = max(ok_t, key=lambda r: r[4])
        print(f"Mejor Tesseract: {best_tess[0]} -> {best_tess[2]}")


if __name__ == "__main__":
    main()
