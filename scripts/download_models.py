"""Descarga (una sola vez) todos los modelos necesarios para trabajar sin Internet.

Uso:  python scripts/download_models.py [--all]
      --all  también descarga AudioLDM y CLIP (opcionales)
"""
import argparse
import shutil
import subprocess
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from storybook.config import load_config  # noqa: E402

cfg = load_config()
ap = argparse.ArgumentParser()
ap.add_argument("--all", action="store_true")
args = ap.parse_args()

# 1. LLM vía Ollama
if shutil.which("ollama"):
    for m in {cfg["llm"]["model"], "llama3.2:3b"}:
        print(f"[ollama] pull {m}")
        subprocess.run(["ollama", "pull", m], check=False)
else:
    print("⚠ Ollama no está instalado: https://ollama.com/download  (luego: ollama pull qwen2.5:3b)")

# 2. Modelos de imagen (quedan en la caché de Hugging Face, ~/.cache/huggingface)
for name, p in cfg["image"]["presets"].items():
    print(f"[imagen:{name}] {p['model_id']}")
    # Solo pesos safetensors en formato diffusers (evita descargar .ckpt/.onnx duplicados)
    snapshot_download(p["model_id"], allow_patterns=["*.json", "*.txt", "*.safetensors"],
                      ignore_patterns=["*.ckpt", "*.onnx", "*non-ema*", "*/*.fp32.*"])
    if p.get("lcm_lora"):
        snapshot_download(p["lcm_lora"])
    if p.get("ip_adapter"):
        hf_hub_download(p["ip_adapter_repo"], f"models/{p['ip_adapter_weight']}")
        snapshot_download(p["ip_adapter_repo"], allow_patterns=["models/image_encoder/*"])

# 3. Voz Piper en español
dest = ROOT / "models" / "piper"
dest.mkdir(parents=True, exist_ok=True)
voice = Path(cfg["tts"]["voice_path"]).stem              # es_MX-claude-high
lang, region, spk, quality = voice.split("_")[0], voice.split("-")[0], voice.split("-")[1], voice.split("-")[2]
for ext in (".onnx", ".onnx.json"):
    f = hf_hub_download("rhasspy/piper-voices", f"{lang}/{region}/{spk}/{quality}/{voice}{ext}")
    shutil.copy(f, dest / f"{voice}{ext}")
print(f"[voz] {dest / voice}.onnx")

# 4. Opcionales
if args.all:
    snapshot_download(cfg["sfx"]["audioldm_model"])
    snapshot_download("openai/clip-vit-base-patch32")

print("\n✔ Listo. A partir de aquí la aplicación funciona sin conexión "
      "(puedes exportar HF_HUB_OFFLINE=1).")
