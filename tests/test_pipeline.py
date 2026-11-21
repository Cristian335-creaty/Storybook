"""Pruebas automáticas (modo simulado, no requieren GPU ni modelos).  pytest -q"""
import json
import wave

import pytest

from storybook.config import load_config
from storybook.pipeline import InputError, StoryPipeline, validate_inputs
from storybook.schema import AMBIENTES
from storybook.sfx import synth
from storybook.story_gen import StoryError, build_story, extract_json, validate_story


@pytest.fixture
def cfg(tmp_path):
    return load_config(overrides={"mock": True, "outputs_dir": str(tmp_path)})


def test_entradas_vacias():
    with pytest.raises(InputError):
        validate_inputs("   ", "terror")
    with pytest.raises(InputError):
        validate_inputs("Un dragón tímido", "")


def test_pipeline_cinco_escenas_completas(cfg):
    story, out = StoryPipeline(cfg, log=lambda m: None).run("Un dragón tímido", "infantil")
    assert len(story.scenes) == 5
    for s in story.scenes:
        assert s.narration and s.image_path and s.voice_path and s.sfx_path
        with wave.open(s.sfx_path) as w:
            assert w.getnframes() > 0
    run = json.loads((out / "run.json").read_text("utf-8"))
    assert run["errors"] == []


def test_validacion_rechaza_cuatro_escenas():
    data = {"title": "x", "characters": [{"name": "A", "visual": "v"}],
            "scenes": [{"narration": "palabra " * 30, "image_prompt": "p"}] * 4}
    with pytest.raises(StoryError, match="exactamente 5"):
        validate_story(data, "terror")


def test_json_con_texto_alrededor_y_ambiente_invalido():
    raw = 'Claro: ```json\n{"title":"T","characters":[{"name":"Ana","visual":"girl"}],' \
          '"scenes":[' + ",".join(
              ['{"narration":"Ana miró el mar y las olas","image_prompt":"p","ambience":"???"}'] * 5) + "]}```"
    d = extract_json(raw)
    story = build_story(d, "idea", "aventura", "style")
    assert all(s.ambience == "mar" for s in story.scenes)      # inferido por palabras clave
    assert story.scenes[0].characters == ["Ana"]               # detectado por mención


@pytest.mark.parametrize("amb", AMBIENTES)
def test_ambientes_audibles(amb):
    y = synth(amb, duration=3, sr=16000, seed=1)
    assert y.shape[0] == 48000 and 0.3 < abs(y).max() <= 0.61


def test_prompt_pone_la_accion_primero_y_es_corto():
    from storybook.image_gen import build_prompt
    from storybook.schema import Character, Scene, Story
    largo = "golden retriever puppy, fluffy golden fur, black harness vest, big brown eyes, " * 3
    chars = [Character(n, "x", largo) for n in ("Bella", "César", "Lola")]
    sc = Scene(1, "t", "texto", "puppy running through a dark park at night", "noche",
               characters=["Bella", "César", "Lola"])
    st = Story("i", "terror", "T", "small town park", "dark gothic illustration, muted colors", chars, [sc])
    p = build_prompt(st, sc)
    assert p.startswith("puppy running through a dark park")
    assert "Lola" not in p                     # máximo 2 personajes
    assert len(p.split()) <= 65                # cabe en el límite de CLIP


def test_escena_corta_se_amplia_sin_descartar_la_historia(cfg):
    """Simula al LLM: la escena 2 viene corta; solo esa escena se amplía."""
    import json as _j
    from storybook.story_gen import StoryGenerator
    larga = "Ana caminó por el bosque oscuro mientras el viento movía las hojas " * 4
    escenas = [{"title": f"E{i}", "narration": larga if i != 2 else "Ana se asustó mucho al oír un ruido extraño detrás de los árboles del bosque.",
                "image_prompt": "girl in forest", "characters": ["Ana"], "ambience": "bosque"}
               for i in range(1, 6)]
    respuestas = iter([
        _j.dumps({"title": "T", "setting": "forest", "scenes": escenas,
                  "characters": [{"name": "Ana", "visual": "girl, red cape"}]}),
        _j.dumps({"narration": "Ana se asustó mucho cuando " + "oyó crujir las ramas " * 12}),
    ])
    cfg["llm"]["strategy"] = "una_llamada"
    cfg["llm"]["unload_after"] = False
    gen = StoryGenerator(cfg)
    gen._chat = lambda messages: next(respuestas)
    story = gen.generate("Ana en el bosque", "fantasía")
    assert len(story.scenes) == 5
    assert len(story.scenes[1].narration.split()) >= 40
    assert story.scenes[0].narration == larga.strip()


def test_detecta_texto_en_ingles():
    from storybook.story_gen import looks_english
    assert looks_english("Luna's curiosity led her deeper into the woods. The forest grew darker")
    assert looks_english("The Hidden Glade")
    assert not looks_english("La curiosidad de Luna la llevó al fondo del bosque oscuro")
    assert not looks_english("Luna y el sauce")
    assert not looks_english("Old Willow")          # un nombre propio no cuenta como inglés


def test_historia_en_ingles_se_rechaza():
    from storybook.story_gen import StoryError, validate_story
    en = "Luna walked into the forest and she saw the old willow with its silver leaves " * 2
    data = {"title": "Luna", "characters": [{"name": "Luna", "visual": "girl"}],
            "scenes": [{"title": "Glade", "narration": en, "image_prompt": "p"}] * 5}
    with pytest.raises(StoryError, match="inglés"):
        validate_story(data, "fantasía")


def test_limpia_restos_de_plantilla():
    from storybook.story_gen import clean
    assert clean("english visual traits: robot negro") == "robot negro"
    assert clean("... RoboBot toma la planta") == "RoboBot toma la planta"
