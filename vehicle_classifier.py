"""
Clasificador de marca/modelo/año del vehículo a partir del crop del coche.
Usa Jordo23/vehicle-classifier (EfficientNet-B4, 8949 clases, Hugging Face).
Opcional: solo se usa si USAR_CLASIFICADOR_MARCA_MODELO = True en la GUI.
Salida solo por consola (log), no se sube al Sheet.
"""

from pathlib import Path

import cv2
import numpy as np

_REPO_ID = "Jordo23/vehicle-classifier"
_FILENAME = "vehicle_classifier.pth"
_INPUT_SIZE = 380
_NUM_CLASSES = 8949

_model_cache = None
_transform_cache = None


def _get_model():
    """Carga el modelo una sola vez (lazy)."""
    global _model_cache, _transform_cache
    if _model_cache is not None:
        return _model_cache, _transform_cache
    try:
        import torch
        import timm
        from torchvision import transforms
        from huggingface_hub import hf_hub_download
    except ImportError as e:
        print(
            "[vehicle_classifier] Para usar marca/modelo instala: pip install torch timm torchvision huggingface_hub",
            file=__import__("sys").stderr,
        )
        raise e
    base_dir = Path(__file__).resolve().parent
    local_path = base_dir / _FILENAME
    if not local_path.is_file():
        try:
            path = hf_hub_download(
                repo_id=_REPO_ID,
                filename=_FILENAME,
                local_dir=base_dir,
                local_dir_use_symlinks=False,
            )
            local_path = Path(path)
        except Exception as e:
            print(f"[vehicle_classifier] Error descargando modelo: {e}", file=__import__("sys").stderr)
            raise
    checkpoint = torch.load(str(local_path), map_location="cpu", weights_only=False)
    model = timm.create_model("efficientnet_b4", pretrained=False, num_classes=_NUM_CLASSES)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    transform = transforms.Compose([
        transforms.Resize((_INPUT_SIZE, _INPUT_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    _model_cache = (model, checkpoint.get("class_mapping"))
    _transform_cache = transform
    return _model_cache, _transform_cache


def predict(car_crop_bgr):
    """
    Clasifica el crop del coche (BGR, numpy).
    Devuelve (etiqueta_str, probabilidad) p. ej. ("Toyota Corolla 2020", 0.45)
    o (None, None) si falla o la imagen no es válida.
    """
    if car_crop_bgr is None or getattr(car_crop_bgr, "size", 0) == 0:
        return None, None
    try:
        import torch
        from PIL import Image
    except ImportError:
        return None, None
    try:
        model_data, transform = _get_model()
        model, class_mapping = model_data
        if not class_mapping:
            return None, None
        # BGR -> RGB, numpy -> PIL
        rgb = cv2.cvtColor(car_crop_bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        tensor = transform(pil).unsqueeze(0)
        with torch.no_grad():
            logits = model(tensor)
            probs = torch.softmax(logits, dim=1)
            top_prob, top_idx = torch.max(probs, dim=1)
        idx = top_idx.item()
        prob = top_prob.item()
        label = class_mapping.get(idx, f"class_{idx}")
        return str(label), float(prob)
    except Exception as e:
        print(f"[vehicle_classifier] Error en predict: {e}", file=__import__("sys").stderr)
        return None, None


def clasificar_y_logear(car_crop_bgr, matricula=None, hora_video=None):
    """
    Ejecuta el clasificador y escribe el resultado en consola (stderr).
    Útil para depuración sin tocar el Sheet.
    """
    label, prob = predict(car_crop_bgr)
    if label is None:
        return
    mat = matricula or "—"
    hora = f" @ {hora_video}" if hora_video else ""
    print(
        f"[Marca/Modelo] Matrícula {mat}{hora} -> {label} ({prob * 100:.1f}%)",
        file=__import__("sys").stderr,
    )
