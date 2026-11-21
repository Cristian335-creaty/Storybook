"""Estructuras de datos compartidas entre los componentes del pipeline.

El objeto ``Story`` es el "contrato" que viaja entre el generador de texto,
el de imágenes, el de voz y el de sonido. Toda la información de consistencia
(personajes, escenario, estilo) vive aquí y se reutiliza en cada escena.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import List, Optional

NUM_ESCENAS = 5

# Conjunto cerrado de ambientes sonoros que el LLM puede elegir.
AMBIENTES = [
    "bosque", "lluvia", "tormenta", "viento", "fuego", "noche", "mar",
    "cueva", "ciudad", "espacio", "suspenso", "magia", "alegre",
]

GENEROS = [
    "fantasía", "terror", "misterio", "aventura", "ciencia ficción", "infantil",
]


@dataclass
class Character:
    name: str
    role: str                 # protagonista, aliado, antagonista...
    visual: str               # descriptores visuales FIJOS (en inglés, para el modelo de imagen)
    personality: str = ""


@dataclass
class Scene:
    index: int                # 1..5
    title: str
    narration: str            # texto narrativo en español (lo que se lee y se narra)
    image_prompt: str         # acción visual de la escena (en inglés)
    ambience: str             # una etiqueta de AMBIENTES
    sfx_prompt: str = ""      # descripción del sonido en inglés (para AudioLDM, opcional)
    characters: List[str] = field(default_factory=list)
    # Rutas de los recursos generados
    image_path: Optional[str] = None
    voice_path: Optional[str] = None
    sfx_path: Optional[str] = None
    errors: List[str] = field(default_factory=list)


@dataclass
class Story:
    idea: str
    genre: str
    title: str
    setting: str              # escenario fijo (inglés, para imagen)
    style: str                # estilo visual fijo (inglés)
    characters: List[Character]
    scenes: List[Scene]
    reference_image: Optional[str] = None   # retrato del protagonista (IP-Adapter)

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "Story":
        return Story(
            idea=d["idea"], genre=d["genre"], title=d["title"],
            setting=d.get("setting", ""), style=d.get("style", ""),
            characters=[Character(**c) for c in d.get("characters", [])],
            scenes=[Scene(**s) for s in d.get("scenes", [])],
            reference_image=d.get("reference_image"),
        )

    def character(self, name: str) -> Optional[Character]:
        for c in self.characters:
            if c.name.lower() == name.lower():
                return c
        return None
