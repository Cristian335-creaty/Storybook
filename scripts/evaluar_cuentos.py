"""Evalúa cuentos YA generados (sin volver a generarlos) y escribe experiments/resultados.csv.

Uso:
  python scripts/evaluar_cuentos.py                 # todos los cuentos reales de outputs/
  python scripts/evaluar_cuentos.py --no-clip       # sin la métrica visual CLIP
  python scripts/evaluar_cuentos.py --desde 20261120-184155 --hasta 20261120-201157

Para cada carpeta de outputs/ con story.json y run.json (excluye las de modo simulado)
calcula: tiempos, palabras por escena, protagonista presente, nombres no declarados,
ambiente respaldado por el texto y, si se puede, similitud CLIP entre escenas.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from storybook.metrics import all_metrics  # noqa: E402
from storybook.pipeline import load_run  # noqa: E402

CONFIG = {"rapido": "A_rapido_prompt+semilla", "consistente": "B_consistente_biblia+ipadapter"}

ap = argparse.ArgumentParser()
ap.add_argument("--no-clip", action="store_true")
ap.add_argument("--desde", default="")
ap.add_argument("--hasta", default="99999999")
args = ap.parse_args()

rows = []
for d in sorted((ROOT / "outputs").iterdir()):
    if not (d / "story.json").exists() or not (d / "run.json").exists():
        continue
    if not (args.desde <= d.name[:15] <= args.hasta + "~"):
        continue
    run = json.loads((d / "run.json").read_text("utf-8"))
    if run.get("mock"):
        continue
    story = load_run(d)
    for s in story.scenes:              # rutas absolutas guardadas en otro equipo/carpeta
        s.image_path = str(d / f"escena_{s.index}.png")
    ref = d / "referencia_protagonista.png"
    story.reference_image = str(ref) if ref.exists() else None
    clip = not args.no_clip
    try:
        m = all_metrics(story, use_clip=clip)
    except Exception as e:              # sin CLIP descargado o sin red: se sigue sin esa métrica
        print(f"  CLIP no disponible ({e}); se omite")
        m, clip = all_metrics(story, use_clip=False), False
    (d / "metricas.json").write_text(json.dumps(m, ensure_ascii=False, indent=2), "utf-8")
    vis = m.get("visual") or {}
    words = [len(s.narration.split()) for s in story.scenes]
    rows.append({
        "carpeta": d.name,
        "config": CONFIG.get(run["image"].get("name"), run["image"].get("name")),
        "estrategia_texto": run["llm"]["strategy"],
        "genero": run["genre"], "idea": run["idea"], "titulo": story.title,
        "personajes": len(story.characters),
        **run["timings"],
        # " de " y " - " en vez de "/": Excel convierte "5/5" en una fecha (5-may)
        "palabras_por_escena": " - ".join(map(str, words)),
        "protagonista_en_escenas": m["narrativa"]["escenas_con_protagonista"].replace("/", " de "),
        "nombres_no_declarados": len(m["narrativa"]["nombres_no_declarados"]),
        "ambiente_respaldado": m["audio"]["ambiente_respaldado_por_texto"].replace("/", " de "),
        "ambientes": " - ".join(m["audio"]["ambientes"]),
        "clip_entre_escenas": vis.get("clip_entre_escenas", ""),
        "clip_vs_referencia": vis.get("clip_vs_referencia", ""),
        "errores": len(run["errors"]),
    })
    print(f"✔ {d.name}: {rows[-1]['config']} · {story.title}")

out = ROOT / "experiments" / "resultados.csv"
out.parent.mkdir(exist_ok=True)
with open(out, "w", newline="", encoding="utf-8-sig") as f:   # utf-8-sig: Excel muestra bien las tildes
    w = csv.DictWriter(f, fieldnames=list(rows[0]), delimiter=";")
    w.writeheader()
    w.writerows(rows)
print(f"\nResultados en {out}")
