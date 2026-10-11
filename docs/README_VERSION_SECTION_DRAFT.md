<!-- Borrador de sección para README.md (no copiar este comentario). La sección ocupa 11 líneas; límite 15. Enlace relativo a la raíz del repo. -->

## Versión E24-E29b: orden numérico (2026-10-11)

La candidata E28-s43 más 15 épocas de mezcla (944 filas de orden + 944 de replay) aprende a comparar el valor del estado con el umbral de la pregunta. Métricas del punto de 15 épocas:

| Medida | testC | testD | mono-hist | E-interp | E-extrap | VE2 | tarea |
|---|---|---|---|---|---|---|---|
| Valor | 0,958 | 0,926 | 0,967 | 0,931 (8/8 puros) | 0,786 (4/8 puros) | 9/10 | 0,690 |

E-extrap 0,786 es el número débil (la candidata intacta tenía 0,857) y la tarea original se recupera desde 0,157. La validación del criterio F1' es formalización pre-registrada por replicación determinista (criterio registrado tras observar E29), no prueba independiente.

Detalles, metodología y reproducción en [docs/RELEASE_NOTES_E24-E29b.md](docs/RELEASE_NOTES_E24-E29b.md).
