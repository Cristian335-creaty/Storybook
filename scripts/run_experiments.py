"""Batería de pruebas de consistencia: 3 casos (idea+género) × 2 configuraciones.

Uso:  python scripts/run_experiments.py [--mock] [--no-clip]

Genera:
  experiments/resultados.csv   tiempos, métricas automáticas y carpeta de cada corrida
  experiments/evaluacion.md    plantilla para la evaluación humana (1-5) por dimensión
"""
import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from storybook.config import load_config  # noqa: E402
from storybook.metrics import all_metrics  # noqa: E402
from storybook.pipeline import StoryPipeline  # noqa: E402

CASOS = [
    ("Una niña descubre que su sombra cobra vida por las noches", "fantasía"),
    ("Un faro abandonado enciende su luz cada noche sin que nadie suba", "terror"),
    ("Un robot de limpieza en una estación espacial encuentra una planta", "ciencia ficción"),
]

CONFIGS = {
    "A_rapido_prompt+semilla": {"image": {"preset": "rapido", "seed_strategy": "variable"},
                                "llm": {"strategy": "una_llamada"}},
    "B_consistente_biblia+ipadapter": {"image": {"preset": "consistente", "seed_strategy": "variable"},
                                       "llm": {"strategy": "biblia_y_escenas"}},
}

ap = argparse.ArgumentParser()
ap.add_argument("--mock", action="store_true")
ap.add_argument("--no-clip", action="store_true")
args = ap.parse_args()

exp = ROOT / "experiments"
exp.mkdir(exist_ok=True)
rows = []
for cname, override in CONFIGS.items():
    cfg = load_config(overrides={**override, "mock": args.mock})
    for idea, genre in CASOS:
        print(f"\n=== {cname} | {genre} | {idea}")
        try:
            story, out = StoryPipeline(cfg).run(idea, genre)
        except Exception as e:     # una prueba fallida se registra y se sigue con las demás
            print(f"  ✘ FALLÓ: {type(e).__name__}: {e}")
            rows.append({"config": cname, "genero": genre, "idea": idea,
                         "titulo": "", "errores": f"FALLO CRÍTICO: {e}", "carpeta": ""})
            continue
        run = json.loads((out / "run.json").read_text("utf-8"))
        m = all_metrics(story, use_clip=not args.no_clip and not args.mock)
        (out / "metricas.json").write_text(json.dumps(m, ensure_ascii=False, indent=2), "utf-8")
        vis = m.get("visual") or {}
        rows.append({
            "config": cname, "genero": genre, "idea": idea, "titulo": story.title,
            **run["timings"],
            "protagonista_en_escenas": m["narrativa"]["escenas_con_protagonista"],
            "nombres_no_declarados": len(m["narrativa"]["nombres_no_declarados"]),
            "ambiente_respaldado": m["audio"]["ambiente_respaldado_por_texto"],
            "clip_entre_escenas": vis.get("clip_entre_escenas", ""),
            "clip_vs_referencia": vis.get("clip_vs_referencia", ""),
            "errores": len(run["errors"]), "carpeta": out.name,
        })

with open(exp / "resultados.csv", "w", newline="", encoding="utf-8") as f:
    fields = max((list(r) for r in rows), key=len)          # la fila más completa define columnas
    w = csv.DictWriter(f, fieldnames=fields, restval="")
    w.writeheader()
    w.writerows(rows)

with open(exp / "evaluacion.md", "w", encoding="utf-8") as f:
    f.write("# Evaluación humana de consistencia (1 = muy mala · 5 = excelente)\n\n")
    f.write("| Config | Género | Narrativa | Visual | Audio y contenido | Funcionamiento | Fallos observados | Cambio realizado |\n")
    f.write("|---|---|---|---|---|---|---|---|\n")
    for r in rows:
        f.write(f"| {r['config']} | {r['genero']} |  |  |  |  |  |  |\n")

print(f"\n✔ Resultados en {exp / 'resultados.csv'}")
