"""Narración con voz sintética local.

Motor principal: Piper (ONNX, rápido en CPU, voces en español).
Respaldo: pyttsx3 (usa SAPI5 en Windows / eSpeak en Linux).
"""
from __future__ import annotations

import subprocess
import sys
import wave
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from .config import resolve

Log = Callable[[str], None]


class PiperTTS:
    def __init__(self, cfg: dict, log: Optional[Log] = None):
        self.cfg = cfg["tts"]
        self.log = log or (lambda m: None)
        self.model = resolve(self.cfg["voice_path"])
        if not self.model.exists():
            raise FileNotFoundError(
                f"No se encontró la voz {self.model}. Ejecuta scripts/download_models.py")
        self.voice = None
        try:
            from piper import PiperVoice
            self.voice = PiperVoice.load(str(self.model))
        except Exception as e:  # la API cambió entre versiones; usamos el CLI
            self.log(f"  Piper (API Python) no disponible: {e}; se usará el CLI")

    def synthesize(self, text: str, out_path: Path) -> str:
        ls = float(self.cfg.get("length_scale", 1.0))
        if self.voice is not None:
            with wave.open(str(out_path), "wb") as wf:
                if hasattr(self.voice, "synthesize_wav"):      # piper-tts >= 1.3
                    from piper import SynthesisConfig
                    self.voice.synthesize_wav(text, wf, syn_config=SynthesisConfig(length_scale=ls))
                else:                                          # piper-tts 1.2
                    self.voice.synthesize(text, wf, length_scale=ls)
        else:
            subprocess.run([sys.executable, "-m", "piper", "--model", str(self.model),
                            "--length_scale", str(ls), "--output_file", str(out_path)],
                           input=text.encode("utf-8"), check=True)
        return str(out_path)


class Pyttsx3TTS:
    def __init__(self, cfg: dict, log: Optional[Log] = None):
        import pyttsx3
        self.engine = pyttsx3.init()
        self.engine.setProperty("rate", 150)
        for v in self.engine.getProperty("voices"):
            if "es" in (v.id + str(getattr(v, "languages", ""))).lower():
                self.engine.setProperty("voice", v.id)
                break

    def synthesize(self, text: str, out_path: Path) -> str:
        self.engine.save_to_file(text, str(out_path))
        self.engine.runAndWait()
        return str(out_path)


class MockTTS:
    """Genera un tono suave de duración proporcional al texto (sin voz real)."""

    def __init__(self, cfg: dict, log: Optional[Log] = None):
        self.sr = 22050

    def synthesize(self, text: str, out_path: Path) -> str:
        dur = max(2.0, len(text.split()) * 0.38)
        t = np.arange(int(dur * self.sr)) / self.sr
        # "sílabas": pulsos de 4 Hz modulando un tono de 220 Hz
        env = 0.5 * (1 + np.sin(2 * np.pi * 4 * t)) * 0.15
        y = env * np.sin(2 * np.pi * 220 * t)
        write_wav(out_path, y, self.sr)
        return str(out_path)


def write_wav(path: Path, y: np.ndarray, sr: int) -> None:
    y = np.clip(y, -1, 1)
    data = (y * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(data.tobytes())


def make_tts(cfg: dict, log: Optional[Log] = None):
    if cfg.get("mock"):
        return MockTTS(cfg, log)
    if cfg["tts"]["engine"] == "piper":
        try:
            return PiperTTS(cfg, log)
        except Exception as e:
            (log or print)(f"  Piper falló ({e}); usando pyttsx3")
    return Pyttsx3TTS(cfg, log)
