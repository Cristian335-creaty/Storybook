"""Generación del texto del cuento con un LLM local (Ollama).

Estrategias (comparables en los experimentos):
  * biblia_y_escenas: 1) se genera la "biblia" (personajes con rasgos fijos,
    escenario y esquema de 5 momentos); 2) se generan las escenas con la biblia
    en el contexto. Mejora la continuidad.
  * una_llamada: todo en un solo prompt (más rápido, menos controlado).

La salida se valida (exactamente 5 escenas, campos obligatorios, ambiente válido)
y, si falla, se reintenta enviando el error al modelo.
"""
from __future__ import annotations

import json
import re
from typing import Callable, Optional

import requests

from . import prompts
from .schema import AMBIENTES, NUM_ESCENAS, Character, Scene, Story

Log = Callable[[str], None]


class StoryError(RuntimeError):
    pass


# ---------------------------------------------------------------- utilidades

def extract_json(text: str) -> dict:
    """Extrae el primer objeto JSON de la respuesta del modelo."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.M).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            raise StoryError("la respuesta no contiene JSON")
        return json.loads(text[start:end + 1])


KEYWORDS = {
    "lluvia": ["lluvia", "llov", "gotas", "aguacero"],
    "tormenta": ["tormenta", "trueno", "rayo", "relámpago"],
    "fuego": ["fuego", "fogata", "llama", "hoguera", "incendio", "chimenea"],
    "mar": ["mar", "ola", "playa", "barco", "océano", "costa"],
    "cueva": ["cueva", "gruta", "túnel", "mina", "sótano"],
    "noche": ["noche", "luna", "estrellas", "grillos", "oscuridad"],
    "espacio": ["nave", "espacio", "planeta", "galaxia", "estación espacial"],
    "ciudad": ["ciudad", "calle", "tráfico", "mercado", "plaza"],
    "magia": ["magia", "hechizo", "brillo", "encantad", "varita", "hada"],
    "viento": ["viento", "montaña", "cima", "desierto", "ventisca"],
    "bosque": ["bosque", "árbol", "selva", "pájaro", "jardín"],
    "suspenso": ["miedo", "sombra", "silencio", "crujido", "acecha"],
    "alegre": ["fiesta", "risa", "celebra", "feliz", "juego"],
}


def mentions(text: str, words) -> bool:
    """True si alguna palabra clave aparece como inicio de palabra (no dentro de otra:
    'mar' no debe coincidir con 'caminar' ni 'amarillo')."""
    t = text.lower()
    return any(re.search(r"(?<![a-záéíóúñ])" + re.escape(w), t) for w in words)


def guess_ambience(text: str, genre: str = "") -> str:
    for amb, words in KEYWORDS.items():
        if mentions(text, words):
            return amb
    return {"terror": "suspenso", "misterio": "suspenso", "infantil": "alegre",
            "ciencia ficción": "espacio", "fantasía": "magia"}.get(genre, "viento")


ES = {"el", "la", "los", "las", "de", "del", "que", "y", "en", "un", "una", "se", "su", "sus",
      "por", "con", "al", "era", "había", "para", "lo", "como", "más", "pero", "muy", "cuando"}
EN = {"the", "and", "of", "to", "was", "were", "her", "his", "she", "he", "in", "with", "that",
      "it", "into", "had", "as", "they", "their", "on", "at", "from", "an", "is"}


def looks_english(text: str) -> bool:
    """Heurística por palabras frecuentes: True si hay más palabras típicas del inglés."""
    words = re.findall(r"[a-záéíóúñü']+", str(text).lower())
    es, en = sum(w in ES for w in words), sum(w in EN for w in words)
    return en > es


MIN_PALABRAS_RECHAZO = 8     # por debajo: JSON roto, se pide de nuevo todo
MIN_PALABRAS_ESCENA = 40     # por debajo: se amplía solo esa escena


def validate_story(data: dict, genre: str) -> None:
    """Lanza StoryError con un mensaje útil para que el LLM se corrija."""
    if not data.get("title"):
        raise StoryError("falta 'title'")
    if looks_english(data["title"]):
        raise StoryError("el título está en inglés; escríbelo en español")
    chars = data.get("characters") or []
    if not chars or not all(c.get("name") and c.get("visual") for c in chars):
        raise StoryError("cada personaje necesita 'name' y 'visual'")
    scenes = data.get("scenes") or []
    if len(scenes) != NUM_ESCENAS:
        raise StoryError(f"hay {len(scenes)} escenas y deben ser exactamente {NUM_ESCENAS}")
    for i, s in enumerate(scenes, 1):
        for key in ("narration", "image_prompt"):
            if not str(s.get(key, "")).strip():
                raise StoryError(f"a la escena {i} le falta '{key}'")
        # Solo se rechaza una narración prácticamente vacía; las escenas cortas
        # no tiran la historia completa: se amplían una por una (ver _expand_short).
        if len(s["narration"].split()) < MIN_PALABRAS_RECHAZO:
            raise StoryError(f"la narración de la escena {i} está casi vacía")
        if looks_english(s["narration"]) or looks_english(s.get("title", "")):
            raise StoryError(f"la escena {i} está en inglés; escribe title y narration en español")


def clean(text: str) -> str:
    """Quita restos de la plantilla que el LLM a veces copia literalmente."""
    t = re.sub(r"^\s*(english visual traits|rasgos visuales)\s*:\s*", "", str(text), flags=re.I)
    return re.sub(r"^\s*(\.\.\.|…)\s*", "", t).strip()


def build_story(data: dict, idea: str, genre: str, style: str) -> Story:
    chars = [Character(name=c["name"].strip(), role=c.get("role", ""),
                       visual=clean(c["visual"]), personality=c.get("personality", ""))
             for c in data["characters"][:3]]
    names = {c.name.lower(): c.name for c in chars}
    scenes = []
    for i, s in enumerate(data["scenes"][:NUM_ESCENAS], 1):
        amb = str(s.get("ambience", "")).strip().lower()
        if amb not in AMBIENTES:
            amb = guess_ambience(s["narration"], genre)
        present = [names[n.lower()] for n in s.get("characters", []) if str(n).lower() in names]
        # Si el LLM no lista personajes, se detectan por mención en el texto
        if not present:
            present = [c.name for c in chars if c.name.lower() in s["narration"].lower()]
        scenes.append(Scene(index=i, title=s.get("title") or f"Escena {i}",
                            narration=clean(s["narration"]),
                            image_prompt=s["image_prompt"].strip(), ambience=amb,
                            sfx_prompt=s.get("sfx_prompt", ""), characters=present))
    return Story(idea=idea, genre=genre, title=data["title"].strip(),
                 setting=data.get("setting", ""), style=style,
                 characters=chars, scenes=scenes)


# ---------------------------------------------------------------- generador

class StoryGenerator:
    def __init__(self, cfg: dict, log: Optional[Log] = None):
        self.cfg = cfg["llm"]
        self.styles = cfg["image"]["style_by_genre"]
        self.log = log or (lambda m: None)

    def _chat(self, messages: list) -> str:
        r = requests.post(
            f"{self.cfg['host']}/api/chat",
            json={"model": self.cfg["model"], "messages": messages, "stream": False,
                  "format": "json",
                  # num_ctx amplio: con el valor por defecto de Ollama los reintentos
                  # (que acumulan mensajes) se truncaban y el modelo perdía las instrucciones.
                  "options": {"temperature": self.cfg["temperature"],
                              "num_ctx": self.cfg.get("num_ctx", 8192)}},
            timeout=self.cfg.get("timeout", 300),
        )
        r.raise_for_status()
        return r.json()["message"]["content"]

    def _ask_json(self, user_prompt: str, check: Callable[[dict], None]) -> dict:
        messages = [{"role": "system", "content": prompts.SYSTEM},
                    {"role": "user", "content": user_prompt}]
        last = None
        for attempt in range(1, self.cfg["max_retries"] + 1):
            raw = self._chat(messages)
            try:
                data = extract_json(raw)
                check(data)
                return data
            except (StoryError, json.JSONDecodeError) as e:
                last = e
                self.log(f"  intento {attempt}: respuesta inválida ({e}); reintentando")
                messages += [{"role": "assistant", "content": raw},
                             {"role": "user", "content": prompts.FIX.format(error=e, n=NUM_ESCENAS)}]
        raise StoryError(f"el LLM no produjo una historia válida: {last}")

    def _expand_short(self, data: dict, genre: str) -> None:
        """Amplía individualmente las escenas con menos de MIN_PALABRAS_ESCENA palabras.

        Los modelos de 3B tienden a acortar escenas cuando escriben las 5 de una vez.
        Pedir de nuevo toda la historia rara vez lo arregla; ampliar solo la escena
        corta, con la biblia y la escena anterior como contexto, sí converge.
        """
        bible = {k: data[k] for k in ("title", "setting", "characters") if k in data}
        for i, s in enumerate(data["scenes"]):
            n = len(s["narration"].split())
            if n >= MIN_PALABRAS_ESCENA:
                continue
            self.log(f"  escena {i + 1}: {n} palabras; ampliándola")
            prev = data["scenes"][i - 1]["narration"] if i else "(es la primera escena)"
            prompt = prompts.expand_prompt(json.dumps(bible, ensure_ascii=False), prev,
                                           s["narration"], genre)

            def check(d):
                if len(str(d.get("narration", "")).split()) < MIN_PALABRAS_ESCENA:
                    raise StoryError("la narración sigue siendo corta: escribe 60 a 90 palabras")
                if looks_english(d["narration"]):
                    raise StoryError("la narración está en inglés; escríbela en español")
            try:
                s["narration"] = self._ask_json(prompt, check)["narration"].strip()
            except StoryError:
                self.log(f"  escena {i + 1}: no se pudo ampliar; se conserva la versión corta")

    def unload(self) -> None:
        """Pide a Ollama que libere el modelo de la VRAM (keep_alive=0)."""
        try:
            requests.post(f"{self.cfg['host']}/api/generate",
                          json={"model": self.cfg["model"], "keep_alive": 0}, timeout=30)
        except requests.RequestException:
            pass

    def generate(self, idea: str, genre: str) -> Story:
        style = self.styles.get(genre, self.styles["default"])
        if self.cfg["strategy"] == "una_llamada":
            data = self._ask_json(prompts.one_shot_prompt(idea, genre),
                                  lambda d: validate_story(d, genre))
        else:
            def check_bible(d):
                if not d.get("title") or not d.get("characters"):
                    raise StoryError("faltan 'title' o 'characters'")
                if looks_english(d["title"]):
                    raise StoryError("el título está en inglés; escríbelo en español")
            bible = self._ask_json(prompts.bible_prompt(idea, genre), check_bible)
            self.log(f"  biblia lista: «{bible['title']}» con "
                     f"{len(bible['characters'])} personaje(s)")
            bible_txt = json.dumps(bible, ensure_ascii=False, indent=1)

            def check_scenes(d):
                validate_story({**bible, "scenes": d.get("scenes")}, genre)
            scenes = self._ask_json(prompts.scenes_prompt(bible_txt, genre), check_scenes)
            data = {**bible, "scenes": scenes["scenes"]}
        self._expand_short(data, genre)
        if self.cfg.get("unload_after"):
            self.unload()
        return build_story(data, idea, genre, style)


# ---------------------------------------------------------------- modo simulado

def mock_story(idea: str, genre: str, cfg: dict) -> Story:
    """Historia de plantilla para probar el pipeline sin LLM."""
    styles = cfg["image"]["style_by_genre"]
    hero = "Lía"
    beats = [
        ("El comienzo", f"En un lugar lejano vivía {hero}, una niña curiosa de capa roja. "
         f"Una tarde escuchó hablar de algo extraordinario: {idea}. Sin pensarlo dos veces, "
         "preparó su mochila, guardó su vieja linterna de bronce y salió por el sendero del "
         "bosque mientras el viento movía las hojas.", "bosque"),
        ("La primera señal", f"{hero} caminó hasta que la noche cubrió el cielo. Entre los "
         "árboles brillaban luces diminutas que parecían llamarla. Los grillos cantaban y la "
         "linterna temblaba en su mano, pero ella recordó por qué había salido y siguió "
         "adelante, paso a paso, siguiendo las luces.", "noche"),
        ("La tormenta", f"De pronto, el cielo rugió y comenzó una tormenta. {hero} corrió a "
         "refugiarse bajo una roca enorme. Allí encontró a Bruno, un zorro gris de bufanda "
         "azul, que también se escondía de la lluvia. Juntos esperaron, compartiendo pan y "
         "contando historias para vencer el miedo.", "tormenta"),
        ("La cueva", f"Cuando cesó la lluvia, Bruno guió a {hero} hasta una cueva antigua. "
         "Las gotas caían del techo y el eco repetía sus pasos. Al fondo, la linterna "
         "iluminó por fin aquello que buscaban, y comprendieron que el verdadero tesoro "
         "estaba en el valor que habían reunido juntos.", "cueva"),
        ("El regreso", f"Al amanecer, {hero} y Bruno volvieron al pueblo. Encendieron una "
         "fogata en la plaza y contaron su aventura a todos. Desde entonces, cada vez que "
         "alguien dudaba en emprender un viaje, la niña de capa roja le prestaba su "
         "linterna de bronce y le recordaba que nunca estaría solo.", "fuego"),
    ]
    chars = [Character(hero, "protagonista",
                       "young girl, short black hair, red hooded cape, brass lantern",
                       "curiosa y valiente"),
             Character("Bruno", "aliado", "small grey fox, blue scarf", "leal y bromista")]
    scenes = [Scene(index=i, title=t, narration=n, image_prompt=f"scene {i}: {t}",
                    ambience=a, characters=[hero] + (["Bruno"] if i >= 3 else []))
              for i, (t, n, a) in enumerate(beats, 1)]
    return Story(idea=idea, genre=genre, title=f"{hero} y la linterna de bronce",
                 setting="enchanted forest village", style=styles.get(genre, styles["default"]),
                 characters=chars, scenes=scenes)
