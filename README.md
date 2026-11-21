# 📖 StoryBook — cuentos ilustrados con IA generativa local

Proyecto Integrador · Electiva IA Generativa · Ingeniería de Sistemas · Universidad Santo Tomás, Tunja (2026)

A partir de **una idea** y **un tipo de historia**, StoryBook genera un cuento de **exactamente 5 escenas**.
Cada escena tiene texto narrativo, ilustración original, narración con voz sintética y ambientación sonora.
Todo se ejecuta **en local**, sin APIs de pago ni servicios de inferencia remotos.

**Integrantes:** Andres Wilches · Cristian Gil · **Docente:** Diego Vela  
**Video de demostración:** [enlace al video]

---

## Arquitectura

```
 idea + género ──► [1] LLM local (Ollama · Qwen2.5-3B)
                       ├─ llamada 1: BIBLIA  (título, personajes con rasgos visuales fijos, escenario, 5 momentos)
                       └─ llamada 2: 5 ESCENAS (narración ES, prompt visual EN, ambiente, personajes)
                       │  validación JSON + reintento con el error → libera VRAM
                       ▼
                 objeto Story (story.json)  ◄── contrato compartido por todos los módulos
                       │
      ┌────────────────┼──────────────────────────┬─────────────────────────┐
      ▼                ▼                          ▼                         ▼
 [2] Imagen       prompt = estilo del género   [3] Voz (Piper, CPU)     [4] Sonido
 SD local         + rasgos fijos personajes     narración de cada        síntesis procedural
 (diffusers)      + acción + escenario          escena → voz_N.wav       propia (13 ambientes)
 retrato ref. →   semilla fija + IP-Adapter                              o AudioLDM → ambiente_N.wav
 escena_N.png
                       ▼
                 [5] Interfaz Streamlit: texto + imagen + 2 reproductores (voz / ambiente en bucle),
                     navegación ◀ ▶ entre las 5 escenas, historial de cuentos generados
```

Los modelos se usan **en secuencia** y se descargan de la VRAM entre etapas, para que quepan en una GPU de 4–6 GB.

### Estrategia de consistencia

| Dimensión | Mecanismo |
|---|---|
| Narrativa | "Biblia" generada primero y reinyectada al generar escenas; validación de 5 escenas; detección de nombres no declarados |
| Visual | El prompt lo arma el código, no el LLM: estilo fijo por género + rasgos fijos de cada personaje presente + escenario; semilla fija; retrato de referencia del protagonista con **IP-Adapter** (preset `consistente`) |
| Audio | Conjunto cerrado de 13 ambientes; si el LLM elige uno inválido se infiere por palabras clave del texto; voz y ambiente en pistas separadas |
| Funcionamiento | Reintento por escena; si una imagen falla se usa un marcador y se registra el error en `run.json` (la app no se cae) |

### Configuraciones a comparar (`scripts/run_experiments.py`)

| | Config A — `rapido` | Config B — `consistente` |
|---|---|---|
| Texto | una llamada | biblia + escenas |
| Imagen | SD-Turbo, 4 pasos, prompt + semilla | DreamShaper 8 (SD1.5) + LCM-LoRA 6 pasos + IP-Adapter (escala 0,35) |
| Semilla | 1234 + n.º de escena | 1234 + n.º de escena |
| Hipótesis | más rápida, rasgos varían | más lenta, protagonista estable |

Métricas automáticas: protagonista presente en las 5 escenas, nombres no declarados, ambiente respaldado por el texto,
similitud **CLIP** entre escenas y contra el retrato de referencia. Se complementan con la plantilla de evaluación humana (1–5).

---

## Resultados de las pruebas

Equipo de prueba: **NVIDIA GeForce RTX 2050 (4,3 GB VRAM) · 8,2 GB RAM · Intel Core 12.ª gen. (Alder Lake) · Windows 11 · Python 3.10.11 · PyTorch 2.6.0 + CUDA 12.4 · Ollama 0.40.16 + qwen2.5:3b.**

Versiones probadas: diffusers 0.41.0 · transformers 5.20.2 · accelerate 1.15.0 · peft 0.21.3 · Streamlit 1.65.0 · piper-tts 1.8.0 · onnxruntime 1.23.2.
Tres ideas (fantasía, terror, ciencia ficción) × dos configuraciones = 6 cuentos completos, sin errores críticos.

| | Config A (rápido) | Config B (consistente) |
|---|---|---|
| Tiempo total promedio por cuento | **178,5 s** (≈ 3 min) | **281,2 s** (≈ 4,7 min) |
| Texto / imágenes / voz (promedio) | 65,9 s / 60,2 s / 51,6 s | 74,2 s / 162,9 s / 43,3 s |
| Evaluación narrativa (1–5) | 3,0 | 3,7 |
| Evaluación visual (1–5) | 2,3 | 4,0 |
| Evaluación audio (1–5) | 3,0 | 3,7 |
| Protagonista presente en el texto | 15/15 escenas | 14/15 escenas |
| Similitud CLIP entre escenas | 0,889 | 0,883 (0,851 frente al retrato) |

**Configuración final: B.** Mantiene reconocible al protagonista en las 5 escenas (en A cambia de aspecto),
a cambio de ~103 s más por cuento. CLIP no distingue entre A y B porque compara la imagen completa
(estilo y escena), no la identidad del personaje; la decisión se basa en la evaluación humana y las láminas de evidencia. Detalle en `experiments/resultados.csv` y `experiments/evaluacion.md`;
capturas en `evidencias/`.

## Requisitos

| | Mínimo (probado como objetivo) | Recomendado |
|---|---|---|
| GPU | NVIDIA 4 GB VRAM (CUDA) — probado en RTX 2050 | NVIDIA 6–8 GB |
| RAM | 8 GB (probado; cerrar otros programas) | 16 GB |
| Disco | ~15 GB para modelos | |
| SO | Windows 10/11 o Linux | |
| Software | Python 3.10–3.12, Ollama, driver NVIDIA con CUDA ≥ 12.4 | |

Sin GPU funciona en CPU, pero cada imagen tarda minutos.

## Instalación paso a paso

```bash
git clone https://github.com/Cristian335-creaty/Storybook.git
cd Storybook
py -3.10 -m venv .venv      # o -3.11 / -3.12
# Windows: .venv\Scripts\activate    ·    Linux: source .venv/bin/activate

# 1) PyTorch con CUDA (ajusta cu124 a tu versión: https://pytorch.org/get-started/locally/; se requiere torch 2.6 o superior)
pip install torch --index-url https://download.pytorch.org/whl/cu124
# 2) Resto de dependencias
pip install -r requirements.txt
# 3) Ollama (https://ollama.com/download) y luego:
python scripts/download_models.py          # LLM, SD, LCM-LoRA, IP-Adapter, voz Piper
python scripts/download_models.py --all    # + AudioLDM y CLIP (opcionales)
```

Ubicación de los modelos: LLM en la carpeta de Ollama; modelos de imagen en la caché de Hugging Face
(`~/.cache/huggingface`); voz Piper en `models/piper/`.

Después de descargar, la app funciona **sin Internet** (opcional: `set HF_HUB_OFFLINE=1` / `export HF_HUB_OFFLINE=1`).

## Ejecución

```bash
streamlit run app.py
```

Ejemplo: idea *"Un faro abandonado enciende su luz cada noche sin que nadie suba"*, tipo *terror* → **Generar cuento**.
Cada cuento queda guardado en `outputs/<fecha>_<idea>/` (imágenes, WAV, `story.json`, `run.json` con tiempos y hardware).

Probar sin modelos (interfaz y flujo):

```bash
set STORYBOOK_MOCK=1          (CMD)   ·   $env:STORYBOOK_MOCK=1 (PowerShell)   ·   export STORYBOOK_MOCK=1 (Linux)
streamlit run app.py
pytest -q                                   # 22 pruebas automáticas
python scripts/run_experiments.py           # genera 3 casos × 2 configuraciones
python scripts/evaluar_cuentos.py           # recalcula métricas (incl. CLIP) de cuentos ya generados
```

## Configuración (`config.yaml`)

`llm.model`, `llm.strategy`, `image.preset`, `image.seed`, `image.seed_strategy`, `tts.voice_path`, `sfx.engine`.
Todas las opciones están comentadas en el archivo.

## Estructura

```
app.py                  interfaz Streamlit
config.yaml             parámetros y presets
storybook/
  schema.py             Story / Scene / Character (contrato entre módulos)
  prompts.py            plantillas del LLM
  story_gen.py          Ollama + validación + reintentos
  image_gen.py          Stable Diffusion + semilla + IP-Adapter
  tts.py                Piper / pyttsx3
  sfx.py                síntesis procedural de ambientes / AudioLDM
  pipeline.py           orquestación, manejo de fallos, registro de tiempos
  metrics.py            métricas de consistencia (narrativa, CLIP, audio)
scripts/                descarga de modelos y batería de experimentos
tests/                  pruebas automáticas (modo simulado)
```

## Modelos y licencias

| Componente | Modelo | Licencia |
|---|---|---|
| Texto | Qwen2.5-3B-Instruct (vía Ollama) | Qwen Research License (uso académico/no comercial) |
| Texto (alternativa) | Llama 3.2 3B | Llama 3.2 Community License |
| Imagen A | stabilityai/sd-turbo | Stability AI Community License |
| Imagen B | Lykon/dreamshaper-8 (base SD 1.5) | CreativeML OpenRAIL-M |
| Imagen B | latent-consistency/lcm-lora-sdv1-5 | OpenRAIL++ |
| Imagen B | h94/IP-Adapter (ip-adapter_sd15) | Apache-2.0 |
| Voz | Piper `es_MX-claude-high` | Apache-2.0 (voz) / MIT (repositorio) |
| Sonido | Síntesis procedural propia (este repo) | MIT |
| Sonido (opcional) | cvssp/audioldm-s-full-v2 | CC BY-NC-SA 4.0 |
| Métrica | openai/clip-vit-base-patch32 | MIT |

Verificar las licencias en la ficha de cada modelo antes de la entrega final. Los pesos **no** se suben al repositorio.

## Consideraciones éticas

- Prompt de sistema y negative prompt que filtran contenido violento o sexual.
- La voz es sintética y no imita a personas reales; se informa al usuario que el audio es generado por IA.
- No se recogen datos personales; todo queda en el equipo local.

## Problemas encontrados y soluciones

| Problema | Causa | Solución |
|---|---|---|
| Error 404 en `/api/chat` | Modelo no descargado en Ollama | `ollama pull qwen2.5:3b` |
| `torch has no attribute 'accelerator'` | PyTorch 2.5 con diffusers reciente | PyTorch ≥ 2.6 (`cu124`) |
| `no attribute 'enable_vae_slicing'` | Cambio de API en diffusers | `pipe.vae.enable_slicing()` con compatibilidad |
| Las 5 imágenes salían iguales | Prompt truncado a 77 tokens (CLIP), IP-Adapter 0,55 y semilla fija | Acción primero en el prompt, máx. 2 personajes, IP-Adapter 0,35, semilla variable |
| Escenas demasiado cortas y cuento descartado | El modelo 3B acorta escenas; contexto truncado en reintentos | Ampliación individual de escenas cortas, `num_ctx` 8192 |
| Cuento en inglés | Rasgos en inglés de la biblia "contagian" al modelo | Instrucción de idioma por campo + validación automática |

## Limitaciones conocidas

- Modelo de 3B parámetros: escenas más largas de lo pedido (hasta ~160 palabras), saltos de continuidad ocasionales y, a veces, prompts de imagen en español.
- IP-Adapter traslada rasgos del protagonista a otros personajes (p. ej., la abuela dibujada como niña) y tiende a repetir la pose del retrato.
- Personajes secundarios pocas veces aparecen en las imágenes (el prompt incluye máximo 2 personajes).
- La métrica de ambiente usa palabras clave: es orientativa, no exacta.
- Con 4 GB de VRAM y 8 GB de RAM se requiere CPU offload; cada imagen de la config B tarda ~27 s.

## Licencia

Código bajo licencia MIT (ver `LICENSE`).
