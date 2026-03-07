"""
Ventana Tkinter: vídeo en tiempo real con detección YOLO + matrículas (PaddleOCR).
Panel derecho: lista con scroll mostrando matrícula y crop de la placa.
"""

import logging
import os
import queue
import sys
import threading
import traceback
import warnings
from pathlib import Path

# Evitar comprobación de conectividad de Paddle y mensajes en consola
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
# Reducir log de Paddle/PaddleOCR para que no salga "Connectivity check..." en cada ejecución
for _ in ("ppocr", "paddle", "paddleocr"):
    logging.getLogger(_).setLevel(logging.WARNING)
# Suprimir avisos de requests (urllib3/chardet) y Paddle en consola
warnings.filterwarnings("ignore", category=UserWarning, module="requests")
warnings.filterwarnings("ignore", message=".*urllib3.*")
warnings.filterwarnings("ignore", message=".*chardet.*")
warnings.filterwarnings("ignore", message=".*DependencyWarning.*")
warnings.filterwarnings("ignore", message=".*pin_memory.*")
# Evitar "unknown argument": Paddle/Ultralytics parsean sys.argv
sys.argv = ["python"]

import io

import cv2
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from PIL import Image, ImageTk

# Import con stderr suprimido para no mostrar mensajes de Paddle en consola
_saved_stderr = sys.stderr
sys.stderr = io.StringIO()
try:
    import deteccion1_funcionando as det
finally:
    sys.stderr = _saved_stderr

# Tamaño máximo del panel de vídeo (redimensionamos el frame para que quepa)
ANCHO_VIDEO = 960
ALTO_VIDEO = 540
# Cada cuántos frames ejecutamos OCR de matrículas (YOLO tracking sigue en todos)
CADA_N_FRAMES = 6
# Frames sin ver un track para considerarlo "salido del ROI" (1–2 s a 25 fps)
FRAMES_PARA_SALIR_ROI = 40
# Solo extraer matrícula si la confianza del vehículo es >= este valor
CONF_MIN_PARA_MATRICULA = 0.5
# Confianza mínima de placa para considerar una matrícula "confirmada" y dejar de intentar leerla
CONF_MATRICULA_CONFIRMADA = 0.80
# Resolución máxima del frame antes de pasarlo a YOLO (None = sin redimensionar)
# Bajar a 480 acelera mucho en GPUs lentas; None mantiene calidad original
YOLO_IMGSZ_MAX = 640
# Tamaño fijo del thumbnail de matrícula en cada tarjeta (todas iguales)
THUMB_ANCHO = 140
THUMB_ALTO = 44
# Altura fija de cada tarjeta de detección
ALTURA_TARJETA = 88
# Ancho del panel derecho
ANCHO_PANEL_LISTA = 260


def frame_a_photoimage(frame_bgr, ancho_max=ANCHO_VIDEO, alto_max=ALTO_VIDEO):
    """Convierte un frame BGR (numpy) a PhotoImage para Tkinter, redimensionando si hace falta."""
    if frame_bgr is None or frame_bgr.size == 0:
        return None
    h, w = frame_bgr.shape[:2]
    if w > ancho_max or h > alto_max:
        escala = min(ancho_max / w, alto_max / h)
        nw, nh = int(w * escala), int(h * escala)
        frame_bgr = cv2.resize(frame_bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    return ImageTk.PhotoImage(pil)


def crop_a_photoimage(crop_bgr, ancho=THUMB_ANCHO, alto=THUMB_ALTO):
    """Convierte el crop de la matrícula a PhotoImage de tamaño fijo (todas iguales)."""
    if crop_bgr is None or crop_bgr.size == 0:
        return None
    h, w = crop_bgr.shape[:2]
    crop_bgr = cv2.resize(crop_bgr, (ancho, alto), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    return ImageTk.PhotoImage(pil)


def dibujar_detecciones(frame, resultados, roi=None, boxes_tracking=None, escala_x=1.0, escala_y=1.0):
    """
    Dibuja en el frame:
    - Zona ROI si está definida (siempre, borde azul)
    - Bboxes de TODOS los coches detectados por YOLO tracking (verde, en cada frame)
    - Bbox de matrícula + texto (cyan, solo cuando hay resultado de OCR y hay ROI)

    boxes_tracking: resultado de YOLO tracking (results_track[0].boxes) para dibujar
                    todos los coches en tiempo real sin esperar al OCR.
    escala_x/y: factores para reescalar coords del frame_yolo al frame original.
    """
    out = frame.copy()

    # --- Zona ROI ---
    if roi is not None:
        rx1, ry1, rx2, ry2 = [int(v) for v in roi]
        cv2.rectangle(out, (rx1, ry1), (rx2, ry2), (128, 128, 255), 2)
        cv2.putText(out, "Zona lectura", (rx1, ry1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (128, 128, 255), 1)

    # --- Bboxes de coches (YOLO corre en ROI; sumar offset para dibujar en frame completo) ---
    rx_off, ry_off = 0, 0
    if roi is not None:
        rx_off, ry_off, _, _ = [int(v) for v in roi]

    # Mapa track_id -> bbox actual (en coords del frame completo) para que las matrículas
    # puedan "seguir" al vehículo aunque el OCR se haya calculado en un frame anterior.
    bbox_por_track = {}
    if boxes_tracking is not None:
        for box in boxes_tracking:
            try:
                bx1, by1, bx2, by2 = box.xyxy[0].tolist()
                bx1, bx2 = bx1 * escala_x + rx_off, bx2 * escala_x + rx_off
                by1, by2 = by1 * escala_y + ry_off, by2 * escala_y + ry_off
                cv2.rectangle(out, (int(bx1), int(by1)), (int(bx2), int(by2)), (0, 255, 0), 2)
                if getattr(box, "id", None) is not None and len(box.id) > 0:
                    tid = int(box.id[0])
                    bbox_por_track[tid] = (bx1, by1, bx2, by2)
            except Exception:
                pass

    # --- Bbox matrícula + texto desde resultados OCR ---
    # Solo dibujar si el track sigue activo en el frame actual
    tracks_activos = set()
    if boxes_tracking is not None:
        for box in boxes_tracking:
            try:
                if getattr(box, "id", None) is not None and len(box.id) > 0:
                    tracks_activos.add(int(box.id[0]))
            except Exception:
                pass

    for r in resultados:
        mat = r.get("matricula")
        if not mat:
            continue
        tid = r.get("track_id")
        if tid is not None and tid >= 0 and tracks_activos and tid not in tracks_activos:
            continue
        pb = r.get("plate_bbox")

        # Si tenemos bbox actual del track, usamos ese para que la caja cyan siga al vehículo.
        # Si no, usamos la plate_bbox calculada por el hilo OCR (coords ya en frame completo).
        if tid is not None and tid in bbox_por_track:
            pb = bbox_por_track[tid]
        if not pb:
            continue
        px1, py1, px2, py2 = [int(v) for v in pb]
        cv2.rectangle(out, (px1, py1), (px2, py2), (0, 255, 255), 2)
        cv2.putText(out, mat, (px1, max(py1 - 5, 15)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    return out


def _redimensionar_para_yolo(frame, max_lado=YOLO_IMGSZ_MAX):
    """Reduce el frame si algún lado supera max_lado, manteniendo proporción."""
    if max_lado is None:
        return frame, 1.0, 1.0
    h, w = frame.shape[:2]
    if w <= max_lado and h <= max_lado:
        return frame, 1.0, 1.0
    escala = max_lado / max(w, h)
    nw, nh = int(w * escala), int(h * escala)
    return cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR), w / nw, h / nh


def worker_video(ruta_video, cada_n, conf_min, conf_min_matricula, modelo_coches, modelo_placas, ocr,
                 cola, stop_event, roi=None):
    """
    Arquitectura de dos hilos:
    - Hilo principal: lee frames + YOLO tracking → detecta tracks que salen del ROI (X frames sin verse)
    - Hilo OCR: procesa boxes y devuelve resultados con car_crop y plate_crop

    Solo se envía "finalizar" (añadir al panel y Sheet) cuando un coche sale del ROI y tiene matrícula.
    """
    import time

    cap = cv2.VideoCapture(str(ruta_video))
    if not cap.isOpened():
        cola.put((None, None, True))
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    tiempo_por_frame = 1.0 / max(fps, 10.0)

    tracks_confirmados = {}
    _ultimos = []
    _pendientes = {}
    cola_ocr = queue.Queue(maxsize=2)

    device = getattr(modelo_coches, "_use_device", None)
    track_kw = {"persist": True, "verbose": False, "conf": conf_min}
    if device is not None:
        track_kw["device"] = device

    def _hilo_ocr():
        while not stop_event.is_set():
            try:
                tarea = cola_ocr.get(timeout=0.5)
            except queue.Empty:
                continue
            if tarea is None:
                break
            frame_detect, frame_full, track_boxes, track_names, ids_a_saltar, escala_x, escala_y, roi_coords = tarea
            try:
                class _FakeResult:
                    def __init__(self, boxes, names):
                        self.boxes = boxes
                        self.names = names
                resultados = det.procesar_cajas_tracked(
                    frame_detect,
                    _FakeResult(track_boxes, track_names),
                    modelo_placas, ocr,
                    conf_min_para_matricula=conf_min_matricula,
                    ids_a_saltar=ids_a_saltar,
                    escala_x=escala_x,
                    escala_y=escala_y,
                )
                for r in resultados:
                    tid = r.get("track_id")
                    c = r.get("plate_conf") or 0.0
                    if tid is not None and tid >= 0 and c >= CONF_MATRICULA_CONFIRMADA:
                        tracks_confirmados[tid] = c
                # Añadir car_crop: desde frame completo con bbox expandido para intentar capturar más coche
                h_fd, w_fd = frame_detect.shape[:2]
                h_f, w_f = frame_full.shape[:2]
                rx1, ry1 = (roi_coords[0], roi_coords[1]) if roi_coords else (0, 0)
                for r in resultados:
                    x1, y1, x2, y2 = r.get("bbox", (0, 0, 0, 0))
                    x1, x2 = max(0, int(x1)), min(w_fd, int(x2))
                    y1, y2 = max(0, int(y1)), min(h_fd, int(y2))
                    if x2 > x1 and y2 > y1:
                        # Convertir a coords frame completo y expandir bbox (30% cada lado)
                        bw, bh = x2 - x1, y2 - y1
                        margin_w, margin_h = max(20, int(bw * 0.35)), max(15, int(bh * 0.35))
                        fx1 = rx1 + x1 - margin_w
                        fy1 = ry1 + y1 - margin_h
                        fx2 = rx1 + x2 + margin_w
                        fy2 = ry1 + y2 + margin_h
                        fx1, fy1 = max(0, fx1), max(0, fy1)
                        fx2, fy2 = min(w_f, fx2), min(h_f, fy2)
                        if fx2 > fx1 and fy2 > fy1:
                            r["car_crop"] = frame_full[fy1:fy2, fx1:fx2].copy()
                        else:
                            r["car_crop"] = frame_detect[y1:y2, x1:x2].copy()
                    else:
                        r["car_crop"] = None
                if roi_coords is not None:
                    rx1, ry1, rx2, ry2 = roi_coords
                    for r in resultados:
                        r["bbox"] = (r["bbox"][0]+rx1, r["bbox"][1]+ry1, r["bbox"][2]+rx1, r["bbox"][3]+ry1)
                        if r.get("plate_bbox"):
                            pb = r["plate_bbox"]
                            r["plate_bbox"] = (pb[0]+rx1, pb[1]+ry1, pb[2]+rx1, pb[3]+ry1)
                _ultimos.clear()
                _ultimos.extend(resultados)
                # Preferir lecturas más largas (placas completas) frente a cortas con más plate_conf
                def _score_lectura(r):
                    pc = r.get("plate_conf") or 0.0
                    mat = (r.get("matricula") or "").strip().upper().replace(" ", "").replace("-", "")
                    return float(pc) * 40.0 + len(mat) * 18.0

                for r in resultados:
                    tid = r.get("track_id")
                    mat = r.get("matricula")
                    if tid is not None and tid >= 0 and mat and (mat or "").strip():
                        prev = _pendientes.get(tid)
                        if prev is None or _score_lectura(r) > _score_lectura(prev):
                            _pendientes[tid] = dict(r)
                # Actualizar panel en tiempo real (como antes)
                con_matricula = [r for r in resultados if r.get("matricula")]
                if con_matricula:
                    cola.put(("detecciones", list(con_matricula), False))
            except Exception:
                pass

    ocr_thread = threading.Thread(target=_hilo_ocr, daemon=True)
    ocr_thread.start()

    frame_idx = 0
    t_ultimo_frame = time.perf_counter()

    try:
        while not stop_event.is_set():
            ret, frame = cap.read()
            if not ret or frame is None:
                cola.put((None, None, True))
                break

            # --- ROI ---
            rx1, ry1 = 0, 0
            if roi is not None:
                rx1, ry1, rx2, ry2 = [int(v) for v in roi]
                h, w = frame.shape[:2]
                rx1, rx2 = max(0, rx1), min(w, rx2)
                ry1, ry2 = max(0, ry1), min(h, ry2)
                frame_detect = frame[ry1:ry2, rx1:rx2] if (rx2 > rx1 and ry2 > ry1) else frame
            else:
                frame_detect = frame

            # --- Tracking YOLO sobre ROI ---
            frame_yolo, escala_x, escala_y = _redimensionar_para_yolo(frame_detect)
            results_track = modelo_coches.track(frame_yolo, **track_kw)
            boxes = results_track[0].boxes if results_track and results_track[0].boxes is not None else None

            # --- Cada N frames: encolar tarea OCR ---
            if frame_idx % cada_n == 0 and boxes is not None:
                ids_a_saltar = set()
                for box in boxes:
                    if getattr(box, "id", None) is not None and len(box.id) > 0:
                        tid = int(box.id[0])
                        if tid in tracks_confirmados:
                            ids_a_saltar.add(tid)
                try:
                    cola_ocr.put_nowait((
                        frame_detect.copy(),
                        frame.copy(),
                        boxes, results_track[0].names,
                        ids_a_saltar, escala_x, escala_y,
                        (rx1, ry1, rx1 + frame_detect.shape[1], ry1 + frame_detect.shape[0]) if roi is not None else None,
                    ))
                except queue.Full:
                    pass

            # --- Detectar tracks que salieron del ROI (X frames sin verse) ---
            current_tracks = set()
            if boxes is not None:
                for box in boxes:
                    if getattr(box, "id", None) is not None and len(box.id) > 0:
                        current_tracks.add(int(box.id[0]))
            for tid in list(_pendientes.keys()):
                if tid in current_tracks:
                    _pendientes[tid]["frames_sin_ver"] = 0
                else:
                    _pendientes[tid]["frames_sin_ver"] = _pendientes[tid].get("frames_sin_ver", 0) + 1
                    if _pendientes[tid]["frames_sin_ver"] >= FRAMES_PARA_SALIR_ROI:
                        r = _pendientes.pop(tid)
                        if r.get("matricula") and (r.get("matricula") or "").strip():
                            # Re-aplicar OCR al crop de la placa al salir: puede dar mejor lectura
                            plate_crop = r.get("plate_crop")
                            if plate_crop is not None and plate_crop.size > 0:
                                try:
                                    nueva_mat, nueva_conf = det.leer_matricula_con_confianza(ocr, plate_crop)
                                    if nueva_mat and (nueva_mat or "").strip():
                                        pc_old = r.get("plate_conf") or 0.0
                                        def _len_norm(m):
                                            return len((m or "").strip().upper().replace(" ", "").replace("-", ""))
                                        score_old = float(pc_old) * 40.0 + _len_norm(r.get("matricula")) * 18.0
                                        score_new = float(nueva_conf) * 40.0 + _len_norm(nueva_mat) * 18.0
                                        if score_new > score_old:
                                            r = dict(r)
                                            r["matricula"] = nueva_mat
                                            r["plate_conf"] = nueva_conf
                                except Exception:
                                    pass
                            cola.put(("finalizar", tid, r))
                        else:
                            pass

            # --- Dibujar: coches (tracking) + matrículas (últimas OCR) ---
            frame_dibujado = dibujar_detecciones(
                frame, _ultimos, roi=roi,
                boxes_tracking=boxes, escala_x=escala_x, escala_y=escala_y,
            )
            cola.put(("frame", frame_dibujado, False))

            # --- Respetar FPS (más lento cuando hay coches en ROI: más tiempo para OCR) ---
            t_ahora = time.perf_counter()
            t_espera = tiempo_por_frame - (t_ahora - t_ultimo_frame)
            # Si hay coches en el ROI, reducir FPS efectivos (~40%) para dar más tiempo al OCR
            coches_en_roi = len(boxes) if boxes is not None else 0
            if coches_en_roi > 0:
                t_espera += tiempo_por_frame * 0.7
            if t_espera > 0:
                time.sleep(t_espera)
            t_ultimo_frame = time.perf_counter()
            frame_idx += 1
    finally:
        cola_ocr.put(None)
        cap.release()


class App:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Detección de vehículos y matrículas")
        self.root.geometry("1280x620")
        self.root.minsize(1000, 500)

        self.modelo_coches = None
        self.modelo_placas = None
        self.ocr = None
        self.cola = queue.Queue()
        self.stop_event = threading.Event()
        self.worker_thread = None
        self.video_ref = None  # mantener referencia a PhotoImage
        self._track_cards = {}
        self._anon_track_counter = 0
        # Contadores globales (persisten entre vídeos)
        self._total_coches = 0
        self._total_matriculas = 0
        self._suma_accuracy = 0.0
        self._imgs_dir = Path(__file__).resolve().parent / "imgs"

        self._construir_ui()
        self._cargar_modelos()

    def _pedir_region_video(self, ruta_video):
        """
        Abre una ventana con el primer fotograma para que el usuario dibuje la región de interés.
        Devuelve (roi, cancelled): roi = (x1,y1,x2,y2) o None para "usar todo"; cancelled = True si canceló.
        """
        cap = cv2.VideoCapture(str(ruta_video))
        if not cap.isOpened():
            messagebox.showerror("Vídeo", "No se pudo abrir el vídeo.")
            return None, True
        ret, frame = cap.read()
        cap.release()
        if not ret or frame is None:
            messagebox.showerror("Vídeo", "No se pudo leer el primer fotograma.")
            return None, True
        self._roi_dialog_result = None
        win = tk.Toplevel(self.root)
        win.title("Definir región de lectura de matrículas")
        win.transient(self.root)
        win.grab_set()
        tk.Label(
            win,
            text="Arrastra el ratón sobre la imagen para dibujar la zona donde leer matrículas (zona más cercana = mejor).",
            fg="#333",
            wraplength=500,
        ).pack(pady=(10, 4))
        # Tamaño máximo de la vista
        max_w, max_h = 920, 520
        h_img, w_img = frame.shape[:2]
        escala = min(max_w / w_img, max_h / h_img, 1.0)
        dw, dh = int(w_img * escala), int(h_img * escala)
        frame_show = cv2.resize(frame, (dw, dh), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(frame_show, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        photo = ImageTk.PhotoImage(pil)
        canvas = tk.Canvas(win, width=dw, height=dh, highlightthickness=1, highlightbackground="#666")
        canvas.pack(padx=10, pady=10)
        canvas.create_image(0, 0, anchor=tk.NW, image=photo)
        canvas.image = photo
        rect_id = [None]
        start_xy = [None]

        def _escala_a_imagen(xc, yc):
            if escala <= 0:
                return 0, 0
            return int(xc / escala), int(yc / escala)

        def _on_down(evt):
            start_xy[0] = (evt.x, evt.y)
            if rect_id[0] is not None:
                canvas.delete(rect_id[0])
            rect_id[0] = canvas.create_rectangle(evt.x, evt.y, evt.x, evt.y, outline="lime", width=2)

        def _on_drag(evt):
            if start_xy[0] is None or rect_id[0] is None:
                return
            x0, y0 = start_xy[0]
            canvas.coords(rect_id[0], x0, y0, evt.x, evt.y)

        def _on_up(evt):
            start_xy[0] = None

        canvas.bind("<ButtonPress-1>", _on_down)
        canvas.bind("<B1-Motion>", _on_drag)
        canvas.bind("<ButtonRelease-1>", _on_up)

        def _aceptar():
            if rect_id[0] is None:
                self._roi_dialog_result = (None, False)
                win.destroy()
                return
            coords = canvas.coords(rect_id[0])
            if len(coords) < 4:
                self._roi_dialog_result = (None, False)
                win.destroy()
                return
            x1c, y1c, x2c, y2c = coords[0], coords[1], coords[2], coords[3]
            x1c, x2c = min(x1c, x2c), max(x1c, x2c)
            y1c, y2c = min(y1c, y2c), max(y1c, y2c)
            if x2c - x1c < 20 or y2c - y1c < 20:
                self._roi_dialog_result = (None, False)
                win.destroy()
                return
            ix1, iy1 = _escala_a_imagen(x1c, y1c)
            ix2, iy2 = _escala_a_imagen(x2c, y2c)
            ix1 = max(0, min(ix1, w_img))
            iy1 = max(0, min(iy1, h_img))
            ix2 = max(0, min(ix2, w_img))
            iy2 = max(0, min(iy2, h_img))
            if ix2 <= ix1 or iy2 <= iy1:
                self._roi_dialog_result = (None, False)
            else:
                self._roi_dialog_result = ((ix1, iy1, ix2, iy2), False)
            win.destroy()

        def _usar_todo():
            self._roi_dialog_result = (None, False)
            win.destroy()

        def _cancelar():
            self._roi_dialog_result = (None, True)
            win.destroy()

        btn_frame = tk.Frame(win)
        btn_frame.pack(pady=8)
        tk.Button(btn_frame, text="Usar todo el frame", command=_usar_todo).pack(side=tk.LEFT, padx=4)
        tk.Button(btn_frame, text="Aceptar región", command=_aceptar).pack(side=tk.LEFT, padx=4)
        tk.Button(btn_frame, text="Cancelar", command=_cancelar).pack(side=tk.LEFT, padx=4)
        win.protocol("WM_DELETE_WINDOW", _cancelar)
        win.wait_window()
        grab_result = getattr(self, "_roi_dialog_result", (None, True))
        return grab_result

    def _construir_ui(self):
        # Barra superior: Abrir vídeo, estado, Prueba (Google Sheet), Cerrar
        barra = tk.Frame(self.root, pady=6)
        barra.pack(side=tk.TOP, fill=tk.X, padx=8)
        tk.Button(barra, text="Abrir vídeo…", command=self._abrir_video).pack(side=tk.LEFT, padx=4)
        self.label_estado = tk.Label(barra, text="Carga modelos y abre un vídeo.", fg="gray")
        self.label_estado.pack(side=tk.LEFT, padx=12)
        self.label_contadores = tk.Label(barra, text="Coches: 0 | Matrículas: 0 | % medio: 0", fg="#555")
        self.label_contadores.pack(side=tk.LEFT, padx=8)
        self._btn_prueba = tk.Button(barra, text="Prueba", command=self._prueba_google_sheet)
        self._btn_prueba.pack(side=tk.RIGHT, padx=4)
        self._btn_prueba.pack_forget()  # Oculto por si se necesita en el futuro
        tk.Button(barra, text="Cerrar", command=self._on_cerrar).pack(side=tk.RIGHT, padx=4)

        # Contenedor principal: vídeo (izq) + lista (der)
        contenido = tk.Frame(self.root, padx=8, pady=8)
        contenido.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Panel izquierdo: vídeo
        marco_video = tk.Frame(contenido)
        marco_video.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.label_video = tk.Label(
            marco_video,
            text="Sin vídeo",
            bg="#1a1a1a",
            fg="#888",
            font=("Segoe UI", 12),
        )
        self.label_video.pack(fill=tk.BOTH, expand=True)

        # Panel derecho: lista con scroll (tarjetas uniformes)
        marco_lista = tk.Frame(contenido, width=ANCHO_PANEL_LISTA)
        marco_lista.pack(side=tk.RIGHT, fill=tk.Y, padx=(12, 0))
        marco_lista.pack_propagate(False)

        titulo_lista = tk.Label(
            marco_lista,
            text="Matrículas detectadas",
            font=("Segoe UI", 11, "bold"),
            fg="#333",
        )
        titulo_lista.pack(anchor=tk.W, pady=(0, 8))
        # Contenedor para lista: Scrollbar a la derecha (siempre visible) y Canvas a la izquierda
        frame_canvas_scroll = tk.Frame(marco_lista)
        frame_canvas_scroll.pack(fill=tk.BOTH, expand=True)
        scrollbar = tk.Scrollbar(frame_canvas_scroll, orient=tk.VERTICAL, width=20)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        canvas = tk.Canvas(
            frame_canvas_scroll,
            highlightthickness=0,
            bg="#eef0f2",
            yscrollcommand=scrollbar.set,
        )
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=canvas.yview)

        self.marco_lista_inner = tk.Frame(canvas, bg="#eef0f2")
        self._canvas_window_id = canvas.create_window((0, 0), window=self.marco_lista_inner, anchor=tk.NW)

        def _actualizar_scrollregion():
            canvas.update_idletasks()
            self.marco_lista_inner.update_idletasks()
            try:
                cw = canvas.winfo_width()
                if cw <= 1:
                    cw = ANCHO_PANEL_LISTA - 35
                h = self.marco_lista_inner.winfo_reqheight()
                w = max(self.marco_lista_inner.winfo_reqwidth(), cw)
                # Altura explícita de la ventana = contenido; así el canvas sabe cuánto hay que scrollar
                canvas.itemconfig(self._canvas_window_id, width=cw, height=max(h, 1))
                canvas.configure(scrollregion=(0, 0, cw, max(h, 1)))
            except tk.TclError:
                pass

        def _on_inner_configure(evt):
            canvas.configure(scrollregion=(0, 0, max(evt.width, 1), max(evt.height, 1)))
            canvas.itemconfig(self._canvas_window_id, height=max(evt.height, 1))

        self.marco_lista_inner.bind("<Configure>", _on_inner_configure)

        def _on_canvas_resize(evt):
            if evt.width > 0:
                canvas.itemconfig(self._canvas_window_id, width=evt.width)
            _actualizar_scrollregion()

        canvas.bind("<Configure>", _on_canvas_resize)

        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        def _on_scroll_linux_up(_):
            canvas.yview_scroll(-2, "units")

        def _on_scroll_linux_down(_):
            canvas.yview_scroll(2, "units")

        canvas.bind("<Enter>", lambda e: canvas.bind_all("<MouseWheel>", _on_mousewheel))
        canvas.bind("<Leave>", lambda e: canvas.unbind_all("<MouseWheel>"))
        canvas.bind("<Button-4>", _on_scroll_linux_up)
        canvas.bind("<Button-5>", _on_scroll_linux_down)

        self._actualizar_scrollregion_lista = _actualizar_scrollregion
        self.canvas_lista = canvas

    def _actualizar_label_contadores(self):
        """Actualiza el label de contadores en la barra."""
        pct = self._suma_accuracy / self._total_matriculas if self._total_matriculas > 0 else 0
        pct_100 = pct * 100
        self.label_contadores.config(
            text=f"Coches: {self._total_coches} | Matrículas: {self._total_matriculas} | % medio: {pct_100:.1f}",
            fg="#555",
        )

    def _cargar_modelos(self):
        self.label_estado.config(text="Cargando modelos.…", fg="orange")
        self.root.update()
        try:
            self.modelo_coches, self.modelo_placas, self.ocr = det.cargar_modelos()
            self.label_estado.config(text="Listo. Abre un vídeo.", fg="green")
        except Exception as e:
            # Escribir traceback en stderr para que salga en consola y en tee salida.txt
            traceback.print_exc(file=sys.stderr)
            sys.stderr.flush()
            self.label_estado.config(text=f"Error cargando modelos: {e}", fg="red")

    def _prueba_google_sheet(self):
        """Prueba la conexión con Google Sheets: añade una fila de prueba y muestra el resultado."""
        try:
            import google_sheet as gs
        except ImportError:
            messagebox.showerror(
                "Google Sheets",
                "No se pudo importar el módulo google_sheet.\n"
                "Asegúrate de que el archivo google_sheet.py está en el mismo directorio que la aplicación."
            )
            return
        self.label_estado.config(text="Probando Google Sheet…", fg="orange")
        self.root.update()
        ok, msg = gs.append_prueba()
        self.label_estado.config(text="Listo. Abre un vídeo." if ok else "Error en prueba Google Sheet.", fg="green" if ok else "gray")
        if ok:
            messagebox.showinfo("Google Sheets – Prueba", msg + "\n\nAbre tu hoja de Google para ver la nueva fila.")
        else:
            messagebox.showerror("Google Sheets – Prueba", msg + "\n\nRevisa GOOGLE_SHEETS_SETUP.md para configurar la conexión.")

    def _abrir_video(self):
        if self.modelo_coches is None or self.modelo_placas is None or self.ocr is None:
            messagebox.showwarning("Modelos", "Espera a que terminen de cargar los modelos.")
            return
        ruta = filedialog.askopenfilename(
            title="Seleccionar vídeo",
            filetypes=[
                ("Vídeo", "*.mp4 *.avi *.mov *.mkv *.webm *.m4v *.wmv"),
                ("Todos", "*.*"),
            ],
        )
        if not ruta:
            return
        self._iniciar_video(Path(ruta))

    def _iniciar_video(self, ruta: Path):
        roi, cancelled = self._pedir_region_video(ruta)
        if cancelled:
            self.label_estado.config(text="Cancelado. Abre otro vídeo si quieres.", fg="gray")
            return
        if self.worker_thread and self.worker_thread.is_alive():
            self.stop_event.set()
            self.worker_thread.join(timeout=2.0)
        self.stop_event.clear()
        self._track_cards = {}
        self._anon_track_counter = 0
        self._limpiar_lista_matriculas()
        self.label_estado.config(text=f"Reproduciendo: {ruta.name}" + (" (solo zona seleccionada)" if roi else ""), fg="blue")
        self.worker_thread = threading.Thread(
            target=worker_video,
            args=(
                ruta,
                CADA_N_FRAMES,
                0.25,
                CONF_MIN_PARA_MATRICULA,
                self.modelo_coches,
                self.modelo_placas,
                self.ocr,
                self.cola,
                self.stop_event,
                roi,
            ),
            daemon=True,
        )
        self.worker_thread.start()
        self._procesar_cola()

    def _procesar_cola(self):
        # Procesar TODOS los mensajes pendientes en la cola para no acumular retraso
        procesados = 0
        while procesados < 10:  # máximo 10 por tick para no congelar la UI
            try:
                msg = self.cola.get_nowait()
            except queue.Empty:
                break
            tipo = msg[0]

            if tipo is None:  # fin de vídeo (formato antiguo por si acaso)
                self.label_estado.config(text="Vídeo terminado. Abre otro si quieres.", fg="gray")
                return

            if tipo == "frame":
                _, frame_dibujado, _ = msg
                self.video_ref = frame_a_photoimage(frame_dibujado)
                if self.video_ref:
                    self.label_video.config(image=self.video_ref, text="")

            elif tipo == "detecciones":
                # Actualización en tiempo real del panel (no contadores ni Sheet)
                _, resultados, _ = msg
                for r in resultados:
                    if not self._tiene_matricula_valida(r):
                        continue
                    track_id = r.get("track_id")
                    mat_norm = self._normalizar_matricula(r.get("matricula") or "")
                    if track_id is not None and track_id >= 0:
                        key = track_id
                    elif mat_norm:
                        existing = self._buscar_track_por_matricula(mat_norm)
                        key = existing if existing is not None else f"_anon_{self._anon_track_counter}"
                        if existing is None:
                            self._anon_track_counter += 1
                    else:
                        key = f"_anon_{self._anon_track_counter}"
                        self._anon_track_counter += 1
                    self._añadir_o_actualizar_matricula(key, r)

            elif tipo == "finalizar":
                _, track_id, r = msg
                if not self._tiene_matricula_valida(r):
                    procesados += 1
                    continue
                # Incrementar contadores (solo contamos coches con matrícula)
                self._total_coches += 1
                self._total_matriculas += 1
                plate_conf = r.get("plate_conf") or 0.0
                self._suma_accuracy += plate_conf
                pct_medio = self._suma_accuracy / self._total_matriculas  # 0-1
                pct_medio_100 = pct_medio * 100  # 0-100 para Sheet
                self._actualizar_label_contadores()
                # Añadir al panel
                key = track_id if track_id is not None and track_id >= 0 else f"_anon_{self._anon_track_counter}"
                if key == f"_anon_{self._anon_track_counter}":
                    self._anon_track_counter += 1
                # Panel: forzamos la lectura definitiva (igual que la del Sheet)
                self._añadir_o_actualizar_matricula(key, r, force_update=True)
                # Subir a Google Sheet y Cloudinary en hebra aparte (evitar bloquear GUI)
                def _subir_en_background():
                    try:
                        import google_sheet as gs
                        car_crop = r.get("car_crop")
                        plate_crop = r.get("plate_crop")
                        # Copias para evitar referencias compartidas en la hebra
                        cc = car_crop.copy() if car_crop is not None and car_crop.size > 0 else None
                        pc = plate_crop.copy() if plate_crop is not None and plate_crop.size > 0 else None
                        gs.append_deteccion(
                            r.get("matricula"),
                            plate_conf,
                            self._total_coches,
                            self._total_matriculas,
                            pct_medio_100,
                            imagen_coche=cc,
                            imagen_placa=pc,
                            carpeta_imgs=str(self._imgs_dir),
                        )
                    except Exception:
                        traceback.print_exc(file=sys.stderr)

                threading.Thread(target=_subir_en_background, daemon=True).start()

            elif tipo is None or (isinstance(msg, tuple) and len(msg) == 3 and msg[2] is True):
                self.label_estado.config(text="Vídeo terminado. Abre otro si quieres.", fg="gray")
                return

            procesados += 1

        if self.worker_thread and self.worker_thread.is_alive():
            self.root.after(16, self._procesar_cola)  # ~60fps polling

    def _tiene_matricula_valida(self, r):
        """True si hay texto de matrícula (no vacío ni guión)."""
        mat = r.get("matricula")
        if mat is None:
            return False
        s = (mat or "").strip().replace("—", "").replace("-", "")
        return len(s) >= 1

    def _normalizar_matricula(self, texto):
        """Texto en mayúsculas sin espacios ni guiones para comparar."""
        if not texto:
            return ""
        return (texto or "").strip().upper().replace(" ", "").replace("-", "").replace("—", "")

    def _calcular_score_lectura(self, r):
        """Mayor = mejor lectura. Favorecemos lecturas más largas (6-7 chars = placa completa)."""
        plate_conf = r.get("plate_conf")
        if plate_conf is None:
            plate_conf = 0.5
        mat = r.get("matricula") or ""
        norm = self._normalizar_matricula(mat)
        # Peso mayor a longitud: una placa completa (6-7 chars) con confianza media
        # debe ganar a una lectura corta (3 chars) con alta confianza
        return float(plate_conf) * 40.0 + len(norm) * 18.0

    def _buscar_track_por_matricula(self, mat_norm):
        """Devuelve el track_id existente cuya matrícula normalizada coincide exactamente, o None."""
        if not mat_norm or mat_norm == "—":
            return None
        for tid, info in self._track_cards.items():
            existente = self._normalizar_matricula(info["best_matricula"])
            if existente and existente == mat_norm:
                return tid
        return None

    def _buscar_track_por_matricula_similar(self, mat_norm):
        """Devuelve track_id si existe una tarjeta con matrícula similar (subcadena). Evita duplicados del mismo coche."""
        if not mat_norm or len(mat_norm) < 3:
            return None
        for tid, info in self._track_cards.items():
            existente = self._normalizar_matricula(info["best_matricula"])
            if not existente or len(existente) < 3:
                continue
            # Misma placa si una es subcadena de la otra (p. ej. BC123 vs ABC123)
            if mat_norm in existente or existente in mat_norm:
                if abs(len(mat_norm) - len(existente)) <= 2:
                    return tid
        return None

    def _añadir_o_actualizar_matricula(self, track_id, r, force_update=False):
        """
        Añade una tarjeta nueva o actualiza la existente.
        Deduplicación doble:
          1. Por track_id (mismo vehículo reconocido por YOLO tracking)
          2. Por texto de matrícula normalizado (mismo texto = mismo coche aunque cambie el ID)
        force_update: si True (usado en "finalizar"), siempre actualiza con este resultado (igual que en el Sheet).
        """
        score = self._calcular_score_lectura(r)
        mat = r.get("matricula") or "—"
        crop = r.get("plate_crop")
        mat_norm = self._normalizar_matricula(mat)

        # 1) Buscar por track_id
        target_id = track_id if track_id in self._track_cards else None

        # 2) Si no hay match por ID, buscar por texto exacto
        if target_id is None:
            target_id = self._buscar_track_por_matricula(mat_norm)
        # 3) Si no hay match exacto, buscar por similar (evitar duplicados: BC123 vs ABC123)
        if target_id is None:
            target_id = self._buscar_track_por_matricula_similar(mat_norm)

        if target_id is not None:
            info = self._track_cards[target_id]
            if not force_update and score <= info["best_score"]:
                # Aunque no mejore el score, unificar el track_id si es diferente
                if target_id != track_id:
                    self._track_cards[track_id] = info
                return
            info["best_score"] = score
            info["best_matricula"] = mat
            info["lbl_mat"].config(text=mat)
            if crop is not None and crop.size > 0:
                thumb = crop_a_photoimage(crop)
                if thumb:
                    info["lbl_img"].config(image=thumb, text="")
                    info["lbl_img"].image = thumb
                    info["img_ref"] = thumb
            # Registrar también bajo el nuevo track_id para futuras búsquedas
            if target_id != track_id:
                self._track_cards[track_id] = info
            self._actualizar_scrollregion_lista()
            self.root.after(150, self._actualizar_scrollregion_lista)
            return

        # Nueva tarjeta (matrícula no vista antes)
        self._crear_tarjeta_matricula(track_id, mat, crop, score)

    def _limpiar_lista_matriculas(self):
        """Destruye todas las tarjetas del panel derecho y resetea el estado interno."""
        for info in self._track_cards.values():
            try:
                info["tarjeta"].destroy()
            except Exception:
                pass
        self._track_cards = {}
        self._anon_track_counter = 0
        self._actualizar_scrollregion_lista()

    def _crear_tarjeta_matricula(self, track_id, mat, crop, score):
        """Crea una nueva tarjeta en el panel y la registra en _track_cards."""
        tarjeta = tk.Frame(
            self.marco_lista_inner,
            height=ALTURA_TARJETA,
            relief=tk.RAISED,
            borderwidth=1,
            bg="#f8f9fa",
            padx=8,
            pady=6,
        )
        tarjeta.pack(anchor=tk.W, fill=tk.X, pady=4)
        tarjeta.pack_propagate(False)

        marco_foto = tk.Frame(tarjeta, width=THUMB_ANCHO, height=THUMB_ALTO, bg="#e9ecef")
        marco_foto.pack(side=tk.TOP, pady=(0, 6))
        marco_foto.pack_propagate(False)
        img_ref = None
        if crop is not None and crop.size > 0:
            thumb = crop_a_photoimage(crop)
            if thumb:
                lbl_img = tk.Label(marco_foto, image=thumb, bg="#e9ecef")
                lbl_img.image = thumb
                lbl_img.place(relx=0.5, rely=0.5, anchor=tk.CENTER)
                img_ref = thumb
        else:
            lbl_img = tk.Label(marco_foto, text="—", bg="#e9ecef", fg="#868e96")
            lbl_img.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

        lbl_mat = tk.Label(
            tarjeta,
            text=mat,
            font=("Consolas", 14, "bold"),
            fg="#212529",
            bg="#f8f9fa",
            height=1,
        )
        lbl_mat.pack(side=tk.TOP, fill=tk.X)

        self._track_cards[track_id] = {
            "tarjeta": tarjeta,
            "lbl_mat": lbl_mat,
            "lbl_img": lbl_img,
            "img_ref": img_ref,
            "best_score": score,
            "best_matricula": mat,
        }
        self._actualizar_scrollregion_lista()
        self.root.after(150, self._actualizar_scrollregion_lista)
        self.root.after(400, self._actualizar_scrollregion_lista)

    def run(self):
        self.root.protocol("WM_DELETE_WINDOW", self._on_cerrar)
        self.root.mainloop()

    def _on_cerrar(self):
        self.stop_event.set()
        # Esperar a que el worker termine antes de destruir (evita crash al cerrar)
        if self.worker_thread and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=3.0)
        try:
            self.root.destroy()
        except Exception:
            pass


def main():
    app = App()
    app.run()


if __name__ == "__main__":
    main()
