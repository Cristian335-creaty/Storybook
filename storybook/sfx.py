"""Ambientación sonora.

Motor por defecto: síntesis procedural propia (numpy/scipy), sin archivos
externos ni licencias de terceros. Cada ambiente combina ruido filtrado,
envolventes y eventos aleatorios (gotas, truenos, chasquidos, pájaros...).
La semilla depende de la escena, así que el resultado es reproducible.

Motor opcional: AudioLDM (difusión texto->audio) a partir de ``sfx_prompt``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

import numpy as np
from scipy import signal

from .schema import AMBIENTES, Scene
from .tts import write_wav

Log = Callable[[str], None]


# ------------------------------------------------------------------ bloques

def _norm(x):
    m = np.max(np.abs(x)) or 1.0
    return x / m


def white(n, rng):
    return rng.standard_normal(n)


def pink(n, rng):
    # filtro de Voss-McCartney aproximado (Paul Kellet)
    b, a = [0.049922035, -0.095993537, 0.050612699, -0.004408786], [1, -2.494956002, 2.017265875, -0.522189400]
    return _norm(signal.lfilter(b, a, white(n, rng)))


def brown(n, rng):
    y = np.cumsum(white(n, rng))
    y = signal.lfilter([1, -1], [1, -0.995], y)   # quita deriva DC
    return _norm(y)


def band(x, lo, hi, sr):
    hi = min(hi, 0.45 * sr)          # nunca por encima de Nyquist
    sos = signal.butter(2, [lo, hi], btype="band", fs=sr, output="sos")
    return signal.sosfilt(sos, x)


def lowpass(x, f, sr):
    return signal.sosfilt(signal.butter(2, f, fs=sr, output="sos"), x)


def lfo(t, f, depth=0.5, phase=0.0):
    return 1 - depth + depth * 0.5 * (1 + np.sin(2 * np.pi * f * t + phase))


def events(n, sr, rate, make, rng):
    """Inserta eventos ``make(rng)`` (arrays cortos) a una tasa de ``rate`` por segundo."""
    out = np.zeros(n)
    k = rng.poisson(rate * n / sr)
    for _ in range(k):
        ev = make(rng)
        start = rng.integers(0, max(1, n - len(ev)))
        out[start:start + len(ev)] += ev[: n - start]
    return out


def chirp(sr, rng):
    d = rng.uniform(0.05, 0.15)
    t = np.arange(int(d * sr)) / sr
    f0, f1 = rng.uniform(2500, 4000), rng.uniform(3000, 6000)
    ph = 2 * np.pi * np.cumsum(np.linspace(f0, f1, len(t))) / sr
    return np.sin(ph) * np.hanning(len(t)) * rng.uniform(0.2, 0.5)


def drop(sr, rng, f=(1500, 3500), d=0.03):
    t = np.arange(int(d * sr)) / sr
    return np.sin(2 * np.pi * rng.uniform(*f) * t) * np.exp(-t * 120) * rng.uniform(0.1, 0.4)


def crackle(sr, rng):
    n = int(rng.uniform(0.002, 0.01) * sr)
    return rng.standard_normal(n) * np.exp(-np.linspace(0, 6, n)) * rng.uniform(0.3, 1.0)


def thunder(sr, rng):
    d = rng.uniform(2.5, 4.0)
    n = int(d * sr)
    t = np.arange(n) / sr
    rumble = lowpass(brown(n, rng), 120, sr) * np.exp(-t * 1.2)
    crack = lowpass(white(n, rng), 1500, sr) * np.exp(-t * 12) * 0.6
    return _norm(rumble + crack) * 0.9


def bell(sr, rng, f=None):
    f = f or rng.choice([1046.5, 1318.5, 1568.0, 2093.0, 2637.0])
    t = np.arange(int(1.5 * sr)) / sr
    y = sum(a * np.sin(2 * np.pi * f * m * t) for a, m in [(1, 1), (0.4, 2.76), (0.2, 5.4)])
    return y * np.exp(-t * 3) * 0.25


def cricket(sr, rng):
    t = np.arange(int(0.25 * sr)) / sr
    gate = (np.sin(2 * np.pi * 30 * t) > 0).astype(float)
    return np.sin(2 * np.pi * 4500 * t) * gate * np.hanning(len(t)) * 0.15


def heartbeat(n, sr, bpm=62):
    out = np.zeros(n)
    beat = int(60 / bpm * sr)
    t = np.arange(int(0.12 * sr)) / sr
    thump = np.sin(2 * np.pi * 55 * t) * np.exp(-t * 35)
    for start in range(0, n - beat, beat):
        for off, amp in ((0, 1.0), (int(0.22 * sr), 0.7)):
            s = start + off
            out[s:s + len(thump)] += thump[: n - s] * amp
    return out


def drone(t, freqs, amp=0.2):
    return sum(np.sin(2 * np.pi * f * t + i) for i, f in enumerate(freqs)) * amp / len(freqs)


def music_box(n, sr, rng):
    scale = [523.3, 587.3, 659.3, 784.0, 880.0, 1046.5]
    out = np.zeros(n)
    step = int(0.35 * sr)
    for i, s in enumerate(range(0, n - step, step)):
        f = scale[[0, 2, 4, 2, 3, 1, 5, 4][i % 8]]
        ev = bell(sr, rng, f) * 0.8
        out[s:s + len(ev)] += ev[: n - s]
    return out


# ------------------------------------------------------------------ ambientes

def synth(ambience: str, duration: float = 12, sr: int = 22050, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = int(duration * sr)
    t = np.arange(n) / sr
    a = ambience if ambience in AMBIENTES else "viento"

    if a == "viento":
        y = band(brown(n, rng), 150, 900, sr) * lfo(t, 0.12, 0.8)
    elif a == "lluvia":
        y = 0.6 * band(pink(n, rng), 800, 8000, sr) + events(n, sr, 25, lambda r: drop(sr, r), rng)
    elif a == "tormenta":
        rain = band(pink(n, rng), 600, 8000, sr) * 0.6
        y = rain + events(n, sr, 0.25, lambda r: thunder(sr, r), rng)
    elif a == "fuego":
        y = 0.5 * lowpass(brown(n, rng), 400, sr) * lfo(t, 0.7, 0.4) + \
            events(n, sr, 18, lambda r: crackle(sr, r), rng)
    elif a == "bosque":
        y = 0.3 * band(brown(n, rng), 200, 1200, sr) * lfo(t, 0.1, 0.6) + \
            events(n, sr, 2.5, lambda r: chirp(sr, r), rng)
    elif a == "noche":
        y = 0.15 * band(brown(n, rng), 100, 600, sr) + \
            events(n, sr, 3.0, lambda r: cricket(sr, r), rng)
    elif a == "mar":
        y = lowpass(pink(n, rng), 1800, sr) * lfo(t, 0.09, 0.9) ** 2
    elif a == "cueva":
        drops = events(n, sr, 1.2, lambda r: drop(sr, r, (600, 1400), 0.08), rng)
        echo = drops.copy()
        for k, g in ((int(0.18 * sr), 0.5), (int(0.41 * sr), 0.3), (int(0.7 * sr), 0.15)):
            echo[k:] += drops[:-k] * g
        y = echo + drone(t, [55, 82.5], 0.15)
    elif a == "ciudad":
        y = 0.5 * lowpass(brown(n, rng), 300, sr) + 0.15 * band(pink(n, rng), 500, 3000, sr) * lfo(t, 0.05, 0.5)
    elif a == "espacio":
        y = drone(t, [55, 55.4, 110.3, 164.8], 0.5) * lfo(t, 0.05, 0.5) + \
            0.05 * band(pink(n, rng), 3000, 6000, sr)
    elif a == "suspenso":
        y = drone(t, [49, 51.9, 98], 0.4) + 0.6 * heartbeat(n, sr) + \
            0.08 * band(brown(n, rng), 100, 400, sr)
    elif a == "magia":
        y = events(n, sr, 3.0, lambda r: bell(sr, r), rng) + drone(t, [261.6, 329.6, 392.0], 0.12) * lfo(t, 0.2, 0.6)
    elif a == "alegre":
        y = music_box(n, sr, rng) + 0.1 * band(brown(n, rng), 200, 1000, sr)

    y = _norm(y) * 0.6
    fade = int(0.5 * sr)                       # fundido para que el bucle no "clique"
    y[:fade] *= np.linspace(0, 1, fade)
    y[-fade:] *= np.linspace(1, 0, fade)
    return y


class ProceduralSFX:
    def __init__(self, cfg: dict, log: Optional[Log] = None):
        self.cfg = cfg["sfx"]

    def generate(self, scene: Scene, out_path: Path) -> str:
        y = synth(scene.ambience, self.cfg["duration"], self.cfg["sample_rate"], seed=scene.index)
        write_wav(out_path, y, self.cfg["sample_rate"])
        return str(out_path)

    def unload(self):
        pass


class AudioLDMSFX:
    """Opcional: genera el ambiente a partir de sfx_prompt con AudioLDM (≈1,5 GB)."""

    def __init__(self, cfg: dict, log: Optional[Log] = None):
        import torch
        from diffusers import AudioLDMPipeline
        self.cfg = cfg["sfx"]
        self.torch = torch
        cuda = torch.cuda.is_available()
        self.pipe = AudioLDMPipeline.from_pretrained(
            self.cfg["audioldm_model"], torch_dtype=torch.float16 if cuda else torch.float32)
        self.pipe.to("cuda" if cuda else "cpu")
        self.fallback = ProceduralSFX(cfg)

    def generate(self, scene: Scene, out_path: Path) -> str:
        if not scene.sfx_prompt:
            return self.fallback.generate(scene, out_path)
        audio = self.pipe(scene.sfx_prompt + ", ambient background, no speech",
                          num_inference_steps=20, audio_length_in_s=self.cfg["duration"],
                          generator=self.torch.Generator("cpu").manual_seed(scene.index)).audios[0]
        write_wav(out_path, _norm(audio) * 0.6, 16000)
        return str(out_path)

    def unload(self):
        self.pipe = None
        if self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()


def make_sfx(cfg: dict, log: Optional[Log] = None):
    if cfg["sfx"]["engine"] == "audioldm" and not cfg.get("mock"):
        try:
            return AudioLDMSFX(cfg, log)
        except Exception as e:
            (log or print)(f"  AudioLDM no disponible ({e}); usando síntesis procedural")
    return ProceduralSFX(cfg, log)
