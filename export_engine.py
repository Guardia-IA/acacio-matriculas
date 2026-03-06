"""
Exporta los modelos YOLO (.pt) a formato TensorRT (.engine) para inferencia más rápida en GPU.

Uso:
  python export_engine.py [--device 0] [--imgsz 640] [--half]

Los .engine se guardan junto a los .pt (o en el directorio del script).
Si ya existe un .engine se puede omitir con --skip-existing.
"""

import argparse
import sys
from pathlib import Path

# Rutas y nombres alineados con deteccion1.py
REPO_PLACAS_HF = "morsetechlab/yolov11-license-plate-detection"
ARCHIVO_PLACAS_HF = "license-plate-finetune-v1n.pt"
NOMBRE_LOCAL_PLACAS = "license-plate-finetune-v1n.pt"
MODELO_COCHES_PT = "yolov8n.pt"


def _obtener_ruta_modelo_placas() -> Path:
    """Ruta al .pt de matrículas; descarga de Hugging Face si no existe."""
    base = Path(__file__).resolve().parent
    local = base / NOMBRE_LOCAL_PLACAS
    if local.is_file():
        return local
    try:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(
            repo_id=REPO_PLACAS_HF,
            filename=ARCHIVO_PLACAS_HF,
            local_dir=base,
            local_dir_use_symlinks=False,
        )
        return Path(path)
    except Exception as e:
        print(f"Error descargando modelo de placas: {e}", file=sys.stderr)
        return local


def export_one(model_path: Path, device: str = "0", imgsz: int = 640, half: bool = False, skip_existing: bool = False) -> bool:
    """Exporta un modelo YOLO a .engine. El .engine se guarda junto al .pt con extensión .engine."""
    out_path = model_path.with_suffix(".engine")
    if skip_existing and out_path.is_file():
        print(f"  [omitido] ya existe: {out_path}")
        return True
    try:
        from ultralytics import YOLO
        model = YOLO(str(model_path))
        # device: "0" o "cuda:0" para GPU; "cpu" para CPU (TensorRT suele requerir GPU)
        model.export(
            format="engine",
            imgsz=imgsz,
            half=half,
            device=device,
            verbose=True,
        )
        print(f"  Exportado: {out_path}")
        return True
    except Exception as e:
        print(f"  Error exportando {model_path}: {e}", file=sys.stderr)
        return False


def main():
    parser = argparse.ArgumentParser(description="Exportar modelos YOLO a TensorRT .engine")
    parser.add_argument("--device", default="0", help="Dispositivo para export (0, cuda:0, cpu)")
    parser.add_argument("--imgsz", type=int, default=640, help="Tamaño de entrada (ej. 640)")
    parser.add_argument("--half", action="store_true", help="Usar FP16 (recomendado en GPU)")
    parser.add_argument("--skip-existing", action="store_true", help="No reexportar si ya existe .engine")
    parser.add_argument("--solo-coches", action="store_true", help="Solo exportar modelo de coches")
    parser.add_argument("--solo-placas", action="store_true", help="Solo exportar modelo de placas")
    args = parser.parse_args()

    base = Path(__file__).resolve().parent

    # Modelo coches: yolov8n.pt -> yolov8n.engine
    coches_pt = base / MODELO_COCHES_PT
    # Modelo placas (mismo directorio que el script o descarga HF)
    placas_pt = _obtener_ruta_modelo_placas()

    ok = True
    if not args.solo_placas:
        if not coches_pt.is_file():
            print(f"No encontrado: {coches_pt}. Descarga yolov8n.pt o especifica ruta.", file=sys.stderr)
            ok = False
        else:
            print("Exportando modelo de coches (YOLO)...")
            ok = export_one(coches_pt, device=args.device, imgsz=args.imgsz, half=args.half, skip_existing=args.skip_existing) and ok

    if not args.solo_coches:
        if not placas_pt.is_file():
            print(f"No encontrado: {placas_pt}. Ejecuta antes la app para que se descargue de Hugging Face.", file=sys.stderr)
            ok = False
        else:
            print("Exportando modelo de placas (YOLO)...")
            ok = export_one(placas_pt, device=args.device, imgsz=args.imgsz, half=args.half, skip_existing=args.skip_existing) and ok

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
