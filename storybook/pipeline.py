"""Orquestador: idea + género -> Story con 5 escenas (texto, imagen, voz, sonido).

Orden pensado para GPU pequeña (los modelos no conviven en VRAM):
  1. LLM (Ollama)      -> texto + biblia      -> se descarga de VRAM
  2. Stable Diffusion  -> referencia + 5 imgs -> se descarga de VRAM
  3. Piper (CPU)       -> 5 narraciones
  4. SFX (CPU)         -> 5 ambientes

Cada ejecución queda registrada en outputs/<id>/ (story.json + run.json con
configuración, hardware, tiempos y errores) como evidencia para las pruebas.
"""
from __future__ import annotations

import json
import platform
import re
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

from .config import image_preset, resolve
from .image_gen import ImageGenerator, MockImageGenerator
from .schema import NUM_ESCENAS, Story
from .sfx import make_sfx
from .story_gen import StoryGenerator, mock_story
from .tts import make_tts

Log = Callable[[str], None]
Progress = Callable[[float, str], None]


class InputError(ValueError):
    pass


def validate_inputs(idea: str, genre: str) -> tuple[str, str]:
    idea, genre = (idea or "").strip(), (genre or "").strip().lower()
    if not idea:
        raise InputError("La idea de la historia no puede estar vacía.")
    if len(idea) < 5:
        raise InputError("Describe la idea con un poco más de detalle (mín. 5 caracteres).")
    if not genre:
        raise InputError("Debes indicar el tipo de historia.")
    return idea, genre


def hardware_info() -> dict:
    info = {"os": platform.platform(), "python": platform.python_version(),
            "cpu": platform.processor() or platform.machine()}
    try:
        import psutil
        info["ram_gb"] = round(psutil.virtual_memory().total / 1e9, 1)
    except ImportError:
        pass
    try:
        import torch
        info["torch"] = torch.__version__
        if torch.cuda.is_available():
            p = torch.cuda.get_device_properties(0)
            info["gpu"] = p.name
            info["vram_gb"] = round(p.total_memory / 1e9, 1)
    except ImportError:
        pass
    return info


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower())[:30].strip("-") or "cuento"


class StoryPipeline:
    def __init__(self, cfg: dict, log: Optional[Log] = None, progress: Optional[Progress] = None):
        self.cfg = cfg
        self.log = log or print
        self.progress = progress or (lambda f, m: None)

    def run(self, idea: str, genre: str) -> tuple[Story, Path]:
        idea, genre = validate_inputs(idea, genre)
        mock = bool(self.cfg.get("mock"))
        run_id = datetime.now().strftime("%Y%m%d-%H%M%S") + "_" + slug(idea)
        out = resolve(self.cfg["outputs_dir"]) / run_id
        out.mkdir(parents=True, exist_ok=True)
        timings, errors = {}, []

        # 1. Texto ------------------------------------------------------------
        self.progress(0.05, "Escribiendo el cuento…")
        t0 = time.time()
        story = mock_story(idea, genre, self.cfg) if mock else \
            StoryGenerator(self.cfg, self.log).generate(idea, genre)
        timings["texto_s"] = round(time.time() - t0, 1)
        self.log(f"Texto: «{story.title}» ({timings['texto_s']} s)")
        self._save(story, out)

        # 2. Imágenes ----------------------------------------------------------
        self.progress(0.25, "Cargando modelo de imagen…")
        t0 = time.time()
        img = MockImageGenerator(self.cfg) if mock else ImageGenerator(self.cfg, self.log)
        fallback = MockImageGenerator(self.cfg)
        try:
            img.load()
            story.reference_image = img.generate_reference(story, out)
        except Exception as e:
            msg = f"no se pudo cargar el modelo de imagen ({type(e).__name__}: {e})"
            errors.append(f"imagen/carga: {msg}")
            print(traceback.format_exc())          # detalle completo en la terminal
            self.log(f"⚠️ {msg}. Se usarán imágenes de respaldo.")
            for s in story.scenes:                 # el aviso aparece en cada escena
                s.errors.append(f"imagen: {msg}")
            img = fallback
        for s in story.scenes:
            self.progress(0.3 + 0.08 * s.index, f"Ilustrando escena {s.index}/{NUM_ESCENAS}…")
            for attempt in (1, 2):
                try:
                    s.image_path = img.generate_scene(story, s, out)
                    break
                except Exception as e:
                    self.log(f"  escena {s.index} intento {attempt} falló: {e}")
                    if attempt == 2:
                        s.errors.append(f"imagen: {e}")
                        s.image_path = fallback.generate_scene(story, s, out)
        img.unload()
        timings["imagenes_s"] = round(time.time() - t0, 1)
        self._save(story, out)

        # 3. Voz ---------------------------------------------------------------
        self.progress(0.75, "Grabando la narración…")
        t0 = time.time()
        tts = make_tts(self.cfg, self.log)
        for s in story.scenes:
            try:
                s.voice_path = tts.synthesize(f"{s.title}. {s.narration}", out / f"voz_{s.index}.wav")
            except Exception as e:
                s.errors.append(f"voz: {e}")
        timings["voz_s"] = round(time.time() - t0, 1)

        # 4. Sonido ------------------------------------------------------------
        self.progress(0.9, "Creando la ambientación sonora…")
        t0 = time.time()
        sfx = make_sfx(self.cfg, self.log)
        for s in story.scenes:
            try:
                s.sfx_path = sfx.generate(s, out / f"ambiente_{s.index}.wav")
            except Exception as e:
                s.errors.append(f"sonido: {e}")
        sfx.unload()
        timings["sonido_s"] = round(time.time() - t0, 1)
        timings["total_s"] = round(sum(timings.values()), 1)

        self._save(story, out)
        run_info = {
            "run_id": run_id, "idea": idea, "genre": genre, "mock": mock,
            "llm": {k: self.cfg["llm"][k] for k in ("model", "strategy", "temperature")},
            "image": {**image_preset(self.cfg), "seed": self.cfg["image"]["seed"],
                      "seed_strategy": self.cfg["image"]["seed_strategy"]},
            "tts": self.cfg["tts"]["engine"], "sfx": self.cfg["sfx"]["engine"],
            "hardware": hardware_info(), "timings": timings,
            "errors": errors + [f"escena {s.index}: {e}" for s in story.scenes for e in s.errors],
        }
        (out / "run.json").write_text(json.dumps(run_info, ensure_ascii=False, indent=2), "utf-8")
        self.progress(1.0, "¡Listo!")
        self.log(f"Terminado en {timings['total_s']} s -> {out}")
        return story, out

    @staticmethod
    def _save(story: Story, out: Path) -> None:
        (out / "story.json").write_text(
            json.dumps(story.to_dict(), ensure_ascii=False, indent=2), "utf-8")


def load_run(folder: Path) -> Story:
    return Story.from_dict(json.loads((Path(folder) / "story.json").read_text("utf-8")))
