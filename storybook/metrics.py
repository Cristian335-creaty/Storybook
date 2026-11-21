"""Métricas automáticas de consistencia (evidencia para el informe).

* Narrativa: el protagonista aparece en todas las escenas, los nombres no
  cambian y no aparecen nombres propios nuevos no declarados en la biblia.
* Visual: similitud coseno CLIP entre las 5 ilustraciones (y contra el retrato
  de referencia si existe). Mayor = personajes/estilo más estables.
* Audio: el ambiente elegido coincide con las palabras clave del texto.

Estas métricas complementan (no reemplazan) la evaluación humana 1-5 de la plantilla.
"""
from __future__ import annotations

import itertools
import re
from typing import Optional

from .schema import Story
from .story_gen import KEYWORDS, mentions

STOP = {"El", "La", "Los", "Las", "Un", "Una", "En", "Al", "Del", "De", "Desde", "Cuando",
        "Allí", "Entre", "Sin", "Juntos", "Cada", "Pero", "Y", "Así", "Luego", "Por", "Con",
        "Era", "Había", "Esa", "Ese", "Este", "Esta", "Aquella", "Aquel", "Mientras", "Tras",
        "Estas", "Estos", "Esas", "Esos", "Su", "Sus", "Ella", "Ellos", "Ellas", "Él", "Nadie",
        "Todo", "Todos", "Algo", "Ahora", "Entonces", "Después", "Antes", "Pronto", "Finalmente",
        "Nunca", "Siempre", "Quizás", "Aunque", "Sin", "Sobre", "Bajo", "Hacia", "Hasta", "Nada"}


def narrative_metrics(story: Story) -> dict:
    hero = story.characters[0].name
    names = {c.name for c in story.characters}
    hero_in = sum(hero.lower() in s.narration.lower() for s in story.scenes)
    unknown = set()
    lower = set(re.findall(r"\b[a-záéíóúñ]{3,}\b", " ".join(s.narration for s in story.scenes)))
    for s in story.scenes:
        for w in re.findall(r"(?<![.!?¡¿]\s)(?<!^)\b([A-ZÁÉÍÓÚÑ][a-záéíóúñ]{2,})", s.narration):
            # palabra que también aparece en minúscula en el cuento = palabra común, no un nombre
            if w not in STOP and w.lower() not in lower and w not in names \
                    and not any(w in n for n in names):
                unknown.add(w)
    return {"protagonista": hero,
            "escenas_con_protagonista": f"{hero_in}/{len(story.scenes)}",
            "nombres_no_declarados": sorted(unknown)}


def audio_relevance(story: Story) -> dict:
    hits = 0
    for s in story.scenes:
        words = KEYWORDS.get(s.ambience, [])
        hits += mentions(s.narration, words)
    return {"ambientes": [s.ambience for s in story.scenes],
            "ambiente_respaldado_por_texto": f"{hits}/{len(story.scenes)}"}


def _features(out):
    """transformers < 5 devuelve un tensor; transformers 5.x devuelve un objeto de salida."""
    import torch
    if torch.is_tensor(out):
        return out
    for key in ("image_embeds", "pooler_output"):
        value = getattr(out, key, None)
        if value is not None:
            return value
    return out[0]


def clip_similarity(story: Story, model_id: str = "openai/clip-vit-base-patch32") -> Optional[dict]:
    try:
        import torch
        from PIL import Image
        from transformers import CLIPModel, CLIPProcessor
    except ImportError:
        return None
    paths = [s.image_path for s in story.scenes if s.image_path]
    model, proc = CLIPModel.from_pretrained(model_id), CLIPProcessor.from_pretrained(model_id)
    with torch.no_grad():
        imgs = [Image.open(p).convert("RGB") for p in paths]
        emb = _features(model.get_image_features(**proc(images=imgs, return_tensors="pt")))
        emb = emb / emb.norm(dim=-1, keepdim=True)
        pairs = [float(emb[i] @ emb[j]) for i, j in itertools.combinations(range(len(paths)), 2)]
        res = {"clip_entre_escenas": round(sum(pairs) / len(pairs), 3)}
        if story.reference_image:
            ref = _features(model.get_image_features(
                **proc(images=[Image.open(story.reference_image).convert("RGB")], return_tensors="pt")))
            ref = ref / ref.norm(dim=-1, keepdim=True)
            res["clip_vs_referencia"] = round(float((emb @ ref.T).mean()), 3)
    return res


def all_metrics(story: Story, use_clip: bool = True) -> dict:
    m = {"narrativa": narrative_metrics(story), "audio": audio_relevance(story)}
    if use_clip:
        m["visual"] = clip_similarity(story)
    return m
