"""Interfaz StoryBook (Streamlit).  Ejecutar:  streamlit run app.py"""
from __future__ import annotations

from pathlib import Path

import streamlit as st

from storybook.config import load_config, resolve
from storybook.pipeline import InputError, StoryPipeline, load_run
from storybook.schema import GENEROS, NUM_ESCENAS

st.set_page_config(page_title="StoryBook IA", page_icon="📖", layout="wide")

if "story" not in st.session_state:
    st.session_state.story, st.session_state.idx = None, 0

cfg = load_config()

# ------------------------------------------------------------------ barra lateral
with st.sidebar:
    st.header("⚙️ Configuración")
    cfg["image"]["preset"] = st.selectbox(
        "Preset de imagen", list(cfg["image"]["presets"]),
        index=list(cfg["image"]["presets"]).index(cfg["image"]["preset"]),
        help="rapido = SD-Turbo (prompt + semilla) · consistente = SD1.5 + IP-Adapter")
    cfg["llm"]["strategy"] = st.selectbox("Estrategia de texto", ["biblia_y_escenas", "una_llamada"])
    cfg["image"]["seed"] = st.number_input("Semilla", value=int(cfg["image"]["seed"]), step=1)
    cfg["mock"] = st.toggle("Modo simulado (sin modelos)", value=bool(cfg.get("mock")))

    st.divider()
    st.subheader("📂 Cuentos anteriores")
    runs = sorted((p for p in resolve(cfg["outputs_dir"]).glob("*") if (p / "story.json").exists()),
                  reverse=True)
    if runs:
        pick = st.selectbox("Abrir", runs, format_func=lambda p: p.name)
        if st.button("Cargar"):
            st.session_state.story, st.session_state.idx = load_run(pick), 0

# ------------------------------------------------------------------ entrada
st.title("📖 StoryBook — cuentos ilustrados con IA local")
with st.form("entrada"):
    idea = st.text_area("Idea de la historia",
                        placeholder="Un faro abandonado que cada noche enciende su luz sin que nadie suba…")
    c1, c2 = st.columns([2, 1])
    genero = c1.selectbox("Tipo de historia", GENEROS + ["otro…"])
    otro = c2.text_input("Si elegiste «otro», escríbelo")
    enviar = st.form_submit_button("✨ Generar cuento", type="primary")

if enviar:
    genre = otro if genero == "otro…" else genero
    bar = st.progress(0.0)
    with st.status("Generando…", expanded=True) as status:
        try:
            pipe = StoryPipeline(cfg, log=st.write, progress=lambda f, m: bar.progress(f, m))
            story, out = pipe.run(idea, genre)
            st.session_state.story, st.session_state.idx = story, 0
            status.update(label=f"Cuento listo · guardado en {out.name}", state="complete")
        except InputError as e:
            status.update(label="Entrada inválida", state="error")
            st.error(str(e))
        except Exception as e:  # error crítico: se muestra sin romper la app
            status.update(label="Error en la generación", state="error")
            st.exception(e)

# ------------------------------------------------------------------ visor de escenas
story = st.session_state.story
if story:
    st.header(story.title)
    st.caption(f"Género: {story.genre} · Personajes: "
               + ", ".join(f"{c.name} ({c.role})" for c in story.characters))
    i = st.session_state.idx
    s = story.scenes[i]

    nav = st.columns([1, 3, 1])
    if nav[0].button("◀ Anterior", disabled=i == 0, width="stretch"):
        st.session_state.idx -= 1
        st.rerun()
    nav[1].markdown(f"<h4 style='text-align:center'>Escena {i + 1} de {NUM_ESCENAS} · {s.title}</h4>",
                    unsafe_allow_html=True)
    if nav[2].button("Siguiente ▶", disabled=i == NUM_ESCENAS - 1, width="stretch"):
        st.session_state.idx += 1
        st.rerun()
    st.progress((i + 1) / NUM_ESCENAS)

    left, right = st.columns([1, 1])
    with left:
        if s.image_path and Path(s.image_path).exists():
            st.image(s.image_path, width="stretch")
    with right:
        st.markdown(f"<p style='font-size:1.15rem;line-height:1.6'>{s.narration}</p>",
                    unsafe_allow_html=True)
        st.markdown("**🗣️ Narración**")
        if s.voice_path:
            st.audio(s.voice_path, autoplay=True)
        st.markdown(f"**🎧 Ambiente:** `{s.ambience}`")
        if s.sfx_path:
            st.audio(s.sfx_path, loop=True, autoplay=True)
        if s.errors:
            st.warning("Fallos recuperados en esta escena: " + "; ".join(s.errors))
