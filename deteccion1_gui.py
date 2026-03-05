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
    import deteccion1 as det
finally:
    sys.stderr = _saved_stderr

# Tamaño máximo del panel de vídeo (redimensionamos el frame para que quepa)
ANCHO_VIDEO = 960
ALTO_VIDEO = 540
# Cada cuántos frames procesamos detección (1 = todos; 5 = cada 5)
CADA_N_FRAMES = 3
# Solo extraer matrícula si la confianza del vehículo es >= este valor (coche más completo)
CONF_MIN_PARA_MATRICULA = 0.5
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


def dibujar_detecciones(frame, resultados, roi=None):
    """Dibuja en el frame los bbox del coche (verde) y de la matrícula (cyan). Si roi, dibuja la zona."""
    out = frame.copy()
    if roi is not None:
        x1, y1, x2, y2 = [int(v) for v in roi]
        cv2.rectangle(out, (x1, y1), (x2, y2), (128, 128, 255), 2)  # Contorno zona de lectura
        cv2.putText(out, "Zona lectura", (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (128, 128, 255), 1)
    for r in resultados:
        x1, y1, x2, y2 = [int(v) for v in r["bbox"]]
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Verde: coche
        cv2.putText(
            out, r["clase"], (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1
        )
        pb = r.get("plate_bbox")
        if pb:
            px1, py1, px2, py2 = [int(v) for v in pb]
            cv2.rectangle(out, (px1, py1), (px2, py2), (255, 255, 0), 2)  # Cyan: matrícula
        mat = r.get("matricula")
        if mat and pb:
            cv2.putText(
                out, mat, (int(pb[0]), int(pb[1]) - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1
            )
    return out


def worker_video(ruta_video, cada_n, conf_min, conf_min_matricula, modelo_coches, modelo_placas, ocr, cola, stop_event, roi=None):
    """
    Lee el vídeo, hace tracking de coches en cada frame y cada N frames extrae matrículas.
    Si roi está definido, solo se procesa esa región. Cada detección lleva track_id para
    que la GUI pueda actualizar la misma matrícula si mejora la lectura.
    """
    cap = cv2.VideoCapture(str(ruta_video))
    if not cap.isOpened():
        cola.put((None, [], True))
        return
    frame_idx = 0
    try:
        while not stop_event.is_set():
            ret, frame = cap.read()
            if not ret or frame is None:
                cola.put((None, [], True))
                break
            if roi is not None:
                x1, y1, x2, y2 = [int(v) for v in roi]
                h, w = frame.shape[:2]
                x1, x2 = max(0, x1), min(w, x2)
                y1, y2 = max(0, y1), min(h, y2)
                frame_detect = frame[y1:y2, x1:x2] if (x2 > x1 and y2 > y1) else frame
            else:
                x1, y1 = 0, 0
                frame_detect = frame

            # Tracking en cada frame para mantener IDs estables
            results_track = modelo_coches.track(
                frame_detect, persist=True, verbose=False, conf=conf_min
            )

            if frame_idx % cada_n == 0 and results_track:
                resultados = det.procesar_cajas_tracked(
                    frame_detect,
                    results_track[0],
                    modelo_placas,
                    ocr,
                    conf_min_para_matricula=conf_min_matricula,
                )
                if roi is not None and x2 > x1 and y2 > y1:
                    for r in resultados:
                        r["bbox"] = (
                            r["bbox"][0] + x1, r["bbox"][1] + y1,
                            r["bbox"][2] + x1, r["bbox"][3] + y1,
                        )
                        if r.get("plate_bbox"):
                            pb = r["plate_bbox"]
                            r["plate_bbox"] = (pb[0] + x1, pb[1] + y1, pb[2] + x1, pb[3] + y1)
                frame_dibujado = dibujar_detecciones(frame, resultados, roi=roi)
                cola.put((frame_dibujado, resultados, False))
            frame_idx += 1
    finally:
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
        # Por track_id: actualizar o añadir tarjeta según si la nueva lectura es mejor
        self._track_cards = {}  # track_id -> {tarjeta, lbl_mat, lbl_img, img_ref, best_score, best_matricula}
        self._anon_track_counter = 0  # para detecciones sin track_id (-1)

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
        # Barra superior: Abrir vídeo, estado, Cerrar
        barra = tk.Frame(self.root, pady=6)
        barra.pack(side=tk.TOP, fill=tk.X, padx=8)
        tk.Button(barra, text="Abrir vídeo…", command=self._abrir_video).pack(side=tk.LEFT, padx=4)
        self.label_estado = tk.Label(barra, text="Carga modelos y abre un vídeo.", fg="gray")
        self.label_estado.pack(side=tk.LEFT, padx=12)
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
        try:
            frame, resultados, es_fin = self.cola.get_nowait()
        except queue.Empty:
            if self.worker_thread and self.worker_thread.is_alive():
                self.root.after(50, self._procesar_cola)
            return
        if es_fin:
            self.label_estado.config(text="Vídeo terminado. Abre otro si quieres.", fg="gray")
            return
        self.video_ref = frame_a_photoimage(frame)
        if self.video_ref:
            self.label_video.config(image=self.video_ref, text="")
        for r in resultados:
            if not self._tiene_matricula_valida(r):
                continue
            track_id = r.get("track_id")
            if track_id is None or track_id < 0:
                track_id = f"_anon_{self._anon_track_counter}"
                self._anon_track_counter += 1
            self._añadir_o_actualizar_matricula(track_id, r)
        self.root.after(50, self._procesar_cola)

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
        """Mayor = mejor lectura (confianza placa + longitud del texto)."""
        plate_conf = r.get("plate_conf")
        if plate_conf is None:
            plate_conf = 0.5
        mat = r.get("matricula") or ""
        norm = self._normalizar_matricula(mat)
        return float(plate_conf) * 100.0 + len(norm)

    def _añadir_o_actualizar_matricula(self, track_id, r):
        """Añade una tarjeta nueva o actualiza la existente si la nueva lectura es mejor."""
        score = self._calcular_score_lectura(r)
        mat = r.get("matricula") or "—"
        crop = r.get("plate_crop")

        if track_id in self._track_cards:
            info = self._track_cards[track_id]
            if score <= info["best_score"]:
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
            self._actualizar_scrollregion_lista()
            self.root.after(150, self._actualizar_scrollregion_lista)
            return

        # Nueva tarjeta
        self._crear_tarjeta_matricula(track_id, mat, crop, score)

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
        self.root.destroy()


def main():
    app = App()
    app.run()


if __name__ == "__main__":
    main()
