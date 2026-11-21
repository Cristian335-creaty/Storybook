# Evaluación de consistencia — StoryBook

Escala: 1 = muy mala · 2 = mala · 3 = aceptable · 4 = buena · 5 = excelente.
Evaluación del equipo (Andres Wilches, Cristian Gil) a partir de la revisión de las 30 ilustraciones y los 6 textos.

Equipo de prueba: NVIDIA GeForce RTX 2050 (4,3 GB VRAM) · 8,2 GB RAM · Windows 11 · Python 3.10.11 · PyTorch 2.6 (CUDA 12.4) · Ollama con qwen2.5:3b.
Parámetros comunes: semilla 1234 (estrategia variable: 1234 + n.º de escena), 512×512, voz Piper es_MX-claude-high, sonido procedural.

| # | Config | Género | Título generado | Narrativa | Visual | Audio y contenido | Funcionamiento | Fallos observados | Cambio realizado / propuesto |
|---|---|---|---|---|---|---|---|---|---|
| 1 | A | fantasía | La Sombra de Elyna | 3 | 2 | 2 | 5 | La niña cambia de color de pelo y edad entre escenas; en la escena 3 aparece una segunda figura. Ambiente «mar» en 3 escenas de un cuento de bosque. La sombra viva aparece hasta la escena 3. | Usar biblia + IP-Adapter (config B). |
| 2 | A | terror | El Faro Único | 3 | 3 | 4 | 5 | El faro es coherente en las 5 imágenes, pero el protagonista (Lucas) nunca aparece. En la escena 4 dice «a la quinta noche» (salto de continuidad). | Rasgos del personaje en el prompt de imagen + IP-Adapter. |
| 3 | A | ciencia ficción | El Robot y la Plantita | 3 | 2 | 3 | 5 | El diseño del robot cambia en cada escena (forma, colores). La planta casi no se ve. Los rasgos visuales se generaron en español. | Biblia con rasgos visuales en inglés + IP-Adapter. |
| 4 | B | fantasía | La Sombra Que Se Transforma | 4 | 4 | 4 | 5 | Luna se mantiene reconocible (rubia, vestido azul y blanco). La Abuela se dibuja como otra niña: IP-Adapter traslada los rasgos de la protagonista a otros personajes. «El Silencioso» nunca aparece en imagen. | Bajar ip_adapter_scale en escenas con varios personajes (trabajo futuro). |
| 5 | B | terror | El Faro Sin Testigos | 4 | 4 | 4 | 5 | Alfredo es idéntico en las 5 escenas y el faro aparece de fondo, pero la composición se repite (retrato frontal centrado en todas). | Equilibrar IP-Adapter (0,35 → 0,25–0,30) para más variedad de pose. |
| 6 | B | ciencia ficción | El Robot y la Plantita | 3 | 4 | 3 | 5 | El robot negro con visor luminoso se mantiene; cambia el color de los ojos (rosa/cian). Escenas empezaban con «...» y los rasgos traían la etiqueta «english visual traits:» copiada de la plantilla. | Limpieza automática de restos de plantilla (implementada). |

## Promedios

| Config | Narrativa | Visual | Audio | Funcionamiento | Tiempo total promedio |
|---|---|---|---|---|---|
| A · rápido (una llamada, SD-Turbo) | 3,0 | 2,3 | 3,0 | 5,0 | 178,5 s (≈ 3 min) |
| B · consistente (biblia, DreamShaper 8 + IP-Adapter) | 3,7 | 4,0 | 3,7 | 5,0 | 281,2 s (≈ 4,7 min) |

## Métrica automática CLIP (similitud coseno, ViT-B/32)

| Prueba | A: entre escenas | B: entre escenas | B: frente al retrato |
|---|---|---|---|
| Fantasía | 0,879 | 0,893 | 0,832 |
| Terror | 0,888 | 0,887 | 0,862 |
| Ciencia ficción | 0,900 | 0,868 | 0,858 |
| **Promedio** | **0,889** | **0,883** | **0,851** |

CLIP no distingue entre A y B: compara la imagen completa (estilo, paleta, tipo de escena) y no la identidad
del personaje. En A el estilo es muy uniforme aunque el protagonista cambie, por eso su similitud es alta.

## Conclusión

Se elige la **configuración B** como final: mejora la consistencia visual de 2,3 a 4,0 y la narrativa de 3,0 a 3,7, a cambio de unos 103 s más por cuento (+58 %), tiempo aceptable para una aplicación local. La configuración A queda como modo rápido para equipos con menos memoria.
