"""Plantillas de prompt para el LLM local."""
from .schema import AMBIENTES, NUM_ESCENAS

SYSTEM = (
    "Eres un escritor de cuentos ilustrados en español. Escribes relatos breves, "
    "coherentes y aptos para todo público (sin violencia gráfica ni contenido sexual). "
    "Respondes SIEMPRE con JSON válido y nada más. "
    "IDIOMA: los campos title, narration, role y personality van SIEMPRE en ESPAÑOL. "
    "Solo visual, setting, image_prompt y sfx_prompt van en inglés (son para el modelo de imagen)."
)

AMB = ", ".join(AMBIENTES)

# --- Estrategia 1: biblia de la historia + escenas (2 llamadas) -------------

BIBLE = """Idea del usuario: "{idea}"
Género: {genre}

Crea la BIBLIA del cuento: título, personajes y escenario, y un esquema de {n} momentos
(inicio, desarrollo 1, desarrollo 2, clímax, desenlace).
Los rasgos visuales de cada personaje deben ser concretos e invariables
(especie/edad, pelo, ropa, colores, accesorio distintivo) y estar EN INGLÉS.

Devuelve este JSON:
{{
  "title": "título EN ESPAÑOL",
  "setting": "escenario principal en inglés, 1 frase",
  "characters": [
    {{"name": "Nombre", "role": "protagonista (en español)", "visual": "english visual traits", "personality": "en español"}}
  ],
  "outline": ["momento 1", "momento 2", "momento 3", "momento 4", "momento 5"]
}}
Máximo 3 personajes. El primero es el protagonista."""

SCENES = """Biblia del cuento (NO cambies nombres, rasgos ni hechos):
{bible}

Escribe exactamente {n} escenas siguiendo el esquema, en orden.
La biblia tiene rasgos en inglés solo para las imágenes: el cuento se escribe en ESPAÑOL.
Cada escena:
- "title": título corto EN ESPAÑOL
- "narration": 60 a 90 palabras EN ESPAÑOL (nunca en inglés), tono de {genre}, continúa la escena anterior
- "image_prompt": EN INGLÉS, qué se ve (acción, lugar, encuadre), máx. 25 palabras, sin rasgos de personajes
- "characters": nombres de los personajes que aparecen
- "ambience": UNA de: {amb}
- "sfx_prompt": EN INGLÉS, el sonido de fondo, máx. 10 palabras

Devuelve: {{"scenes": [ ... {n} objetos ... ]}}"""

# --- Estrategia 2: todo en una llamada --------------------------------------

ONE_SHOT = """Idea del usuario: "{idea}"
Género: {genre}

Escribe un cuento de exactamente {n} escenas (inicio, desarrollo, clímax y desenlace).
Devuelve este JSON:
{{
  "title": "título EN ESPAÑOL",
  "setting": "escenario en inglés",
  "characters": [{{"name": "...", "role": "...", "visual": "english visual traits", "personality": "..."}}],
  "scenes": [
    {{"title": "...", "narration": "60-90 palabras EN ESPAÑOL", "image_prompt": "english, max 25 words",
      "characters": ["..."], "ambience": "una de: {amb}", "sfx_prompt": "english, max 10 words"}}
  ]
}}"""

EXPAND = """Cuento de {genre}. Datos fijos (no cambies nombres ni rasgos):
{bible}

Escena anterior: {prev}

Esta escena quedó demasiado corta: "{scene}"

Reescríbela en español con 60 a 90 palabras: conserva los hechos y los nombres,
agrega detalles de lo que ven, oyen y sienten los personajes, y que continúe la escena anterior.
Devuelve: {{"narration": "..."}}"""

FIX = """Tu respuesta anterior no es válida: {error}
Corrígela y devuelve SOLO el JSON completo con exactamente {n} escenas."""


def bible_prompt(idea, genre):
    return BIBLE.format(idea=idea, genre=genre, n=NUM_ESCENAS)


def scenes_prompt(bible_json, genre):
    return SCENES.format(bible=bible_json, genre=genre, n=NUM_ESCENAS, amb=AMB)


def expand_prompt(bible_json, prev, scene, genre):
    return EXPAND.format(bible=bible_json, prev=prev, scene=scene, genre=genre)


def one_shot_prompt(idea, genre):
    return ONE_SHOT.format(idea=idea, genre=genre, n=NUM_ESCENAS, amb=AMB)
