"""Generación de ilustraciones con Stable Diffusion local (diffusers).

Estrategia de consistencia visual:
  1. Prompt compuesto en código (no por el LLM): estilo fijo del género +
     rasgos fijos de los personajes presentes + acción de la escena + escenario.
  2. Semilla fija para todas las escenas (seed_strategy=fija).
  3. (Preset "consistente") Se genera primero un retrato de referencia del
     protagonista y se usa como imagen condicionante con IP-Adapter en las 5 escenas.

Optimización para 4-6 GB de VRAM: fp16, CPU offload, VAE slicing, 512x512,
pocos pasos (SD-Turbo o LCM-LoRA).
"""
from __future__ import annotations

import gc
import hashlib
import textwrap
from pathlib import Path
from typing import Callable, Optional

from PIL import Image, ImageDraw

from .config import image_preset
from .schema import Scene, Story

Log = Callable[[str], None]


def _words(text: str, n: int) -> str:
    return " ".join(str(text).replace("\n", " ").split()[:n])


def build_prompt(story: Story, scene: Scene) -> str:
    """Prompt de la escena, ordenado por prioridad.

    CLIP (el codificador de texto de SD) solo lee ~77 tokens: lo que pase de ahí
    se descarta. Por eso va PRIMERO la acción de la escena (lo que cambia entre
    escenas), luego los rasgos fijos de hasta 2 personajes presentes (resumidos),
    el escenario y al final el estilo. Así la acción nunca se pierde.
    """
    names = scene.characters or [story.characters[0].name]
    traits = []
    for name in names[:2]:
        c = story.character(name)
        if c:
            traits.append(f"{c.name}: {_words(c.visual, 12)}")
    parts = [_words(scene.image_prompt, 22), "; ".join(traits),
             _words(story.setting, 8), _words(story.style, 10)]
    return ", ".join(p for p in parts if p)


def build_reference_prompt(story: Story) -> str:
    hero = story.characters[0]
    return (f"{story.style}, full body character portrait of {hero.visual}, "
            "front view, neutral pose, plain light background, character design sheet")


class ImageGenerator:
    def __init__(self, cfg: dict, log: Optional[Log] = None):
        self.cfg = cfg["image"]
        self.preset = image_preset(cfg)
        self.log = log or (lambda m: None)
        self.pipe = None
        self.torch = None

    # -------------------------------------------------------------- carga
    def load(self):
        import torch
        from diffusers import AutoPipelineForText2Image, LCMScheduler

        self.torch = torch
        p = self.preset
        cuda = torch.cuda.is_available()
        dtype = torch.float16 if cuda else torch.float32
        self.log(f"  cargando {p['model_id']} ({'GPU' if cuda else 'CPU'})")
        pipe = AutoPipelineForText2Image.from_pretrained(
            p["model_id"], torch_dtype=dtype,
            variant="fp16" if cuda and "turbo" in p["model_id"] else None,
            safety_checker=None,
        )
        if p.get("lcm_lora"):
            pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
            pipe.load_lora_weights(p["lcm_lora"])
            pipe.fuse_lora()
        if p.get("ip_adapter"):
            pipe.load_ip_adapter(p["ip_adapter_repo"], subfolder="models",
                                 weight_name=p["ip_adapter_weight"])
        if cuda and self.cfg.get("cpu_offload", True):
            pipe.enable_model_cpu_offload()
        elif cuda:
            pipe.to("cuda")
        # Decodificar el VAE por partes ahorra VRAM. La API cambió entre versiones de diffusers:
        if hasattr(pipe.vae, "enable_slicing"):        # diffusers recientes
            pipe.vae.enable_slicing()
        elif hasattr(pipe, "enable_vae_slicing"):      # diffusers <= 0.3x
            pipe.enable_vae_slicing()
        pipe.set_progress_bar_config(disable=True)
        self.pipe = pipe

    def unload(self):
        self.pipe = None
        gc.collect()
        if self.torch is not None and self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()

    # -------------------------------------------------------------- generación
    def _seed(self, index: int) -> int:
        base = int(self.cfg["seed"])
        return base if self.cfg.get("seed_strategy", "fija") == "fija" else base + index

    def _run(self, prompt: str, seed: int, ref: Optional[Image.Image], ip_scale: float):
        p = self.preset
        kwargs = dict(prompt=prompt, num_inference_steps=p["steps"],
                      guidance_scale=p["guidance"], width=p["width"], height=p["height"],
                      generator=self.torch.Generator("cpu").manual_seed(seed))
        if p["guidance"] > 1.0:
            kwargs["negative_prompt"] = self.cfg["negative_prompt"]
        if p.get("ip_adapter"):
            self.pipe.set_ip_adapter_scale(ip_scale)
            kwargs["ip_adapter_image"] = ref or Image.new("RGB", (224, 224), "white")
        return self.pipe(**kwargs).images[0]

    def generate_reference(self, story: Story, out_dir: Path) -> Optional[str]:
        if not self.preset.get("ip_adapter"):
            return None
        img = self._run(build_reference_prompt(story), self._seed(0), None, 0.0)
        path = out_dir / "referencia_protagonista.png"
        img.save(path)
        return str(path)

    def generate_scene(self, story: Story, scene: Scene, out_dir: Path) -> str:
        ref = Image.open(story.reference_image) if story.reference_image else None
        img = self._run(build_prompt(story, scene), self._seed(scene.index), ref,
                        self.preset.get("ip_adapter_scale", 0.0))
        path = out_dir / f"escena_{scene.index}.png"
        img.save(path)
        return str(path)


# -------------------------------------------------------------- modo simulado

class MockImageGenerator:
    """Dibuja tarjetas de color para probar la interfaz sin Stable Diffusion."""

    def __init__(self, cfg: dict, log: Optional[Log] = None):
        self.preset = {"name": "mock", "ip_adapter": False}

    def load(self):
        pass

    def unload(self):
        pass

    def generate_reference(self, story, out_dir):
        return None

    def generate_scene(self, story: Story, scene: Scene, out_dir: Path) -> str:
        h = hashlib.md5(story.title.encode()).digest()
        base = (h[0], h[1], h[2])
        top = tuple(min(255, int(c * 0.5 + 40 * scene.index)) for c in base)
        img = Image.new("RGB", (512, 512), top)
        d = ImageDraw.Draw(img)
        for y in range(512):
            k = y / 512
            d.line([(0, y), (512, y)],
                   fill=tuple(int(t * (1 - k) + 20 * k) for t in top))
        d.ellipse([196, 220, 316, 340], fill=(200, 40, 40))       # "protagonista"
        d.text((20, 20), f"[MOCK] Escena {scene.index}: {scene.title}", fill="white")
        y = 380
        for line in textwrap.wrap(build_prompt(story, scene), 70)[:8]:
            d.text((20, y), line, fill=(230, 230, 230))
            y += 14
        path = out_dir / f"escena_{scene.index}.png"
        img.save(path)
        return str(path)
