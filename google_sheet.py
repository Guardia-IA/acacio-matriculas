"""
Integración con Google Sheets para registrar matrículas detectadas.

Para la prueba se usa solo texto. Más adelante se puede añadir subida de imágenes
a Drive y enlace en la hoja (columna "Imagen coche", "Imagen matrícula").

Configuración:
  - GOOGLE_CREDENTIALS_JSON: ruta al JSON de la cuenta de servicio (p. ej. google_credentials.json).
  - GOOGLE_SHEET_URL o GOOGLE_SHEET_NAME: URL de la hoja o nombre del documento.
    Si usas nombre, la primera hoja del documento se usa por defecto.
"""

import os
import random
import string
import sys
import traceback
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

# Ruta por defecto al JSON de la cuenta de servicio (en el directorio del proyecto)
_BASE_DIR = Path(__file__).resolve().parent
DEFAULT_CREDENTIALS_PATH = _BASE_DIR / "google_credentials.json"

# URL de la hoja (ej: https://docs.google.com/spreadsheets/d/ID_DE_LA_HOJA/edit)
# o nombre del documento si prefieres abrir por nombre.
# Puedes dejar estos valores fijos o sobreescribirlos con variables de entorno.
GOOGLE_SHEET_URL = os.environ.get(
    "GOOGLE_SHEET_URL",
    "https://docs.google.com/spreadsheets/d/1fjtneiUVt8APOT62Wr8IzA7USrGaHg7TgrVufzhypiU/edit?gid=0#gid=0",
)
GOOGLE_SHEET_NAME = os.environ.get("GOOGLE_SHEET_NAME", "Matrículas detectadas")

# Configuración de Cloudinary:
# Formato de CLOUDINARY_URL: cloudinary://API_KEY:API_SECRET@CLOUD_NAME
# OJO: aquí va SOLO la parte 'cloudinary://...' (sin el prefijo 'CLOUDINARY_URL=').
CLOUDINARY_URL = os.environ.get(
    "CLOUDINARY_URL",
    "cloudinary://824173837123264:D2hQvrRjkxh212PROfy5XEYCqdo@dihmr5bhq",
)
# Carpeta (en Cloudinary) donde se guardarán las imágenes de coches/matrículas.
CLOUDINARY_FOLDER = os.environ.get("CLOUDINARY_FOLDER", "matriculas")


def _get_client(credentials_path=None):
    """Devuelve el cliente gspread autenticado con cuenta de servicio."""
    try:
        import gspread
        from google.oauth2.service_account import Credentials
    except ImportError as e:
        raise ImportError(
            "Faltan dependencias para Google Sheets. Instala:\n"
            "  pip install gspread google-auth"
        ) from e

    path = credentials_path or os.environ.get("GOOGLE_CREDENTIALS_JSON") or str(DEFAULT_CREDENTIALS_PATH)
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"No se encuentra el archivo de credenciales: {path}\n"
            "Crea una cuenta de servicio en Google Cloud y descarga el JSON. "
            "Ver GOOGLE_SHEETS_SETUP.md."
        )

    scopes = [
        "https://www.googleapis.com/auth/spreadsheets",
    ]
    creds = Credentials.from_service_account_file(str(path), scopes=scopes)
    return gspread.authorize(creds)


def _get_sheet(client):
    """Abre la hoja por URL o por nombre."""
    if GOOGLE_SHEET_URL.strip():
        return client.open_by_url(GOOGLE_SHEET_URL.strip()).sheet1
    if GOOGLE_SHEET_NAME.strip():
        return client.open(GOOGLE_SHEET_NAME.strip()).sheet1
    raise ValueError(
        "Configura GOOGLE_SHEET_URL o GOOGLE_SHEET_NAME "
        "(o variables de entorno GOOGLE_SHEET_URL / GOOGLE_SHEET_NAME)."
    )


# Fila donde se escribirán las cabeceras y a partir de la cual se añadirá información.
# Ejemplo: HEADERS_ROW = 9 -> los datos reales empezarán en la fila 10.
HEADERS_ROW = 9


def ensure_headers(sheet, headers, row: int = HEADERS_ROW):
    """Si la fila indicada está vacía o no coincide, escribe la cabecera en esa fila."""
    try:
        current = sheet.row_values(row)
    except Exception:
        current = []
    # Solo comprobamos el prefijo hasta len(headers) para no sobrescribir columnas extra
    if not current or current[: len(headers)] != headers:
        sheet.update(f"A{row}", [headers], value_input_option="RAW")


# Cabeceras que usaremos al integrar con detección (imágenes como enlace o embebidas).
# Solo las 5 columnas que forman la tabla de log; los resúmenes se mantienen en B7/D7/F7.
HEADERS_DETECCION = [
    "Fecha y hora",
    "Matrícula",
    "% acierto (placa)",
    "Imagen coche",
    "Imagen matrícula",
]


def _upload_image_cloudinary(img_bgr, public_id_prefix: str):
    """
    Sube un array numpy BGR a Cloudinary y devuelve la URL segura.
    Requiere:
      - pip install cloudinary
      - variable de entorno CLOUDINARY_URL (o cloud_name/api_key/api_secret)
    """
    if img_bgr is None:
        return None
    try:
        import cv2
        import numpy as np  # noqa: F401
        import cloudinary
        import cloudinary.uploader
    except ImportError as e:
        print(f"[Cloudinary] ImportError: {e}", file=sys.stderr)
        return None

    if img_bgr.size == 0:
        print("[Cloudinary] Imagen vacía, no se sube.", file=sys.stderr)
        return None

    ok, buf = cv2.imencode(".jpg", img_bgr)
    if not ok:
        print("[Cloudinary] Error al codificar la imagen a JPG.", file=sys.stderr)
        return None

    # Configurar Cloudinary a partir de CLOUDINARY_URL
    if CLOUDINARY_URL:
        print(f"[Cloudinary] Usando CLOUDINARY_URL: {CLOUDINARY_URL}", file=sys.stderr)
        try:
            parsed = urlparse(CLOUDINARY_URL)
            # netloc: api_key:api_secret@cloud_name
            creds, cloud_name = parsed.netloc.split("@")
            api_key, api_secret = creds.split(":", 1)
            cloudinary.config(
                cloud_name=cloud_name,
                api_key=api_key,
                api_secret=api_secret,
                secure=True,
            )
            print(
                f"[Cloudinary] Config OK. cloud_name={cloud_name}, api_key={api_key}",
                file=sys.stderr,
            )
        except Exception as e:
            print(f"[Cloudinary] Error parseando CLOUDINARY_URL: {e}", file=sys.stderr)
            cloudinary.config(secure=True)
    else:
        print("[Cloudinary] CLOUDINARY_URL vacío; usando config por defecto.", file=sys.stderr)
        cloudinary.config(secure=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    rand = "".join(random.choices(string.ascii_lowercase + string.digits, k=6))
    public_id = f"{public_id_prefix}_{ts}_{rand}"

    try:
        print(f"[Cloudinary] Subiendo imagen con public_id='{public_id}', folder='{CLOUDINARY_FOLDER}'", file=sys.stderr)
        res = cloudinary.uploader.upload(
            buf.tobytes(),
            folder=CLOUDINARY_FOLDER,
            public_id=public_id,
            overwrite=False,
            resource_type="image",
        )
        url = res.get("secure_url") or res.get("url")
        print(f"[Cloudinary] Subida OK. URL={url}", file=sys.stderr)
        return url
    except Exception:
        print("[Cloudinary] Error al subir la imagen:", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        return None


def _subir_imagen_dummy(nombre_base: str, color_bgr, credentials_path=None) -> str:
    """
    Crea una imagen simple en memoria (rectángulo de color) y la sube a Cloudinary.
    Si falla la subida o Cloudinary no está configurado, devuelve None.
    """
    try:
        import cv2
        import numpy as np
    except ImportError:
        return None

    img = np.zeros((120, 200, 3), dtype=np.uint8)
    img[:, :] = color_bgr
    return _upload_image_cloudinary(img, nombre_base)


def append_prueba(credentials_path=None):
    """
    Añade una fila de prueba a la hoja.
    Devuelve (True, "mensaje") en éxito o (False, "mensaje de error").
    """
    try:
        client = _get_client(credentials_path)
        sheet = _get_sheet(client)
        ensure_headers(sheet, HEADERS_DETECCION)

        # Leer métricas de la parte superior de la hoja
        def _to_int(val: str) -> int:
            try:
                return int(float(val.replace(",", ".")))
            except Exception:
                return 0

        def _to_float(val: str) -> float:
            try:
                return float(val.replace(",", "."))
            except Exception:
                return 0.0

        coches_prev = _to_int(sheet.acell("B7").value or "")
        mats_prev = _to_int(sheet.acell("D7").value or "")
        acc_prev = _to_float(sheet.acell("F7").value or "")

        # Datos aleatorios de prueba (una nueva detección de matrícula)
        ahora = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        matricula_fake = "TEST-" + "".join(random.choices(string.ascii_uppercase + string.digits, k=4))
        pct_val = random.uniform(70, 99)
        pct_fake = f"{pct_val:.1f}"

        # Imágenes dummy: verde para coche, azul para matrícula
        url_coche = _subir_imagen_dummy("coche_prueba", (0, 255, 0), credentials_path)
        url_placa = _subir_imagen_dummy("matricula_prueba", (255, 0, 0), credentials_path)

        # Fórmula IMAGE() para que se vea la imagen embebida en la celda.
        # Si la subida falla o Cloudinary no está configurado, dejamos la celda vacía.
        if url_coche:
            img_coche_cell = f'=IMAGE("{url_coche}")'
        else:
            img_coche_cell = ""
        if url_placa:
            img_placa_cell = f'=IMAGE("{url_placa}")'
        else:
            img_placa_cell = ""

        row = [ahora, matricula_fake, pct_fake, img_coche_cell, img_placa_cell]
        sheet.append_row(row, value_input_option="USER_ENTERED")

        # Actualizar los resúmenes B7 / D7 / F7:
        #   - coches visualizados: +1
        #   - matrículas detectadas: +1
        #   - % medio: media simple entre el valor previo y el nuevo
        coches_nuevo = coches_prev + 1
        mats_nuevo = mats_prev + 1
        if coches_prev <= 0:
            acc_nuevo = pct_val
        else:
            acc_nuevo = (acc_prev * coches_prev + pct_val) / coches_nuevo
        sheet.update_acell("B7", str(coches_nuevo))
        sheet.update_acell("D7", str(mats_nuevo))
        sheet.update_acell("F7", f"{acc_nuevo:.1f}")

        return True, (
            "Fila de prueba añadida correctamente.\n"
            f"Fecha/hora: {ahora}\n"
            f"Matrícula: {matricula_fake}\n"
            f"Coches (B7): {coches_nuevo}, Matrículas (D7): {mats_nuevo}, % global (F7): {acc_nuevo:.1f}"
        )
    except FileNotFoundError as e:
        return False, str(e)
    except ValueError as e:
        return False, str(e)
    except ImportError as e:
        return False, str(e)
    except Exception as e:
        return False, f"Error al escribir en la hoja: {e}\n\nComprueba que la hoja esté compartida con el email de la cuenta de servicio."


def append_deteccion(matricula, confianza_placa, total_coches, total_matriculas, imagen_coche=None, imagen_placa=None, credentials_path=None):
    """
    Añade una fila con una detección real (para integrar con el flujo de vídeo).
    imagen_coche e imagen_placa: arrays numpy BGR o None; si se pasan, por ahora
    se puede guardar como "Sí" y más adelante subir a Drive y poner el enlace.
    """
    try:
        client = _get_client(credentials_path)
        sheet = _get_sheet(client)
        ensure_headers(sheet, HEADERS_DETECCION)

        ahora = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        pct = f"{float(confianza_placa) * 100:.1f}" if confianza_placa is not None else ""

        # Para la integración real podríamos subir las imágenes igual que en la prueba;
        # de momento solo marcamos si existen.
        img_coche = "Sí" if imagen_coche is not None else "N/A"
        img_placa = "Sí" if imagen_placa is not None else "N/A"

        row = [ahora, matricula or "", pct, img_coche, img_placa]
        sheet.append_row(row, value_input_option="USER_ENTERED")
        return True
    except Exception:
        return False
