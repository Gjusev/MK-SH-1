# Notas de versión E24-E29b: orden numérico en el modelo de decisiones tipadas (2026-10-11)

## Qué es esta versión

La versión E24-E29b es la candidata E28-s43 (sha256 75ea9eed…414829, tensor verificado durante la revisión de E29) más 15 épocas de mezcla: en cada época, 944 filas del microconjunto de orden numérico junto a 944 filas de replay de las tareas originales. El modelo sigue siendo JuliaArch-small (mmBERT-small con cabeza de decisión) y conserva lo anterior (decisión calibrada con abstención sobre las tareas originales); lo nuevo es la capacidad de comparar el valor del estado con el umbral de la pregunta, es decir, orden numérico real y no atajos.

Receta de la candidata: 944 filas (rejilla de 92 valores × 8 umbrales, más 16 valores extremos con la misma rejilla y 8 cadenas de monotonía; balance 472/472), parada por dev-paciencia-10 y finalista igual al checkpoint final, sin selección posterior.

## Métricas completas

Punto de 15 épocas (la versión):

| Medida | Valor |
|---|---|
| testC (orden, conocido) | 0,958 (46/48) |
| testD (interpolación histórica) | 0,926, intacto |
| mono-hist (monotonía histórica) | 0,967, intacto |
| E-interp (OOD, dos decimales) | 0,931 (108/116), 8/8 puros |
| E-extrap (OOD, dos decimales) | 0,786 (44/56), 4/8 puros |
| VE2 (monotonía OOD) | 9/10 (único fallo: barrido 118,57) |
| Tarea original | 0,690 (207/300), recuperada desde 0,157 |

Referencias para leer la tabla:

| Punto | testC | testD | mono-hist | E-interp | E-extrap | VE2 | Tarea |
|---|---|---|---|---|---|---|---|
| Candidata E28-s43 intacta | 1,000 | 0,926 | 0,967 | 0,911 (testE global) | 0,857 | 9/10 | 0,157 |
| Mezcla, 15 épocas (esta versión) | 0,958 | 0,926 | 0,967 | 0,931 (8/8) | 0,786 (4/8) | 9/10 | 0,690 |
| Mezcla, 20 épocas | 0,958 | 0,926 | 0,967 | 0,897 (7/8) | 0,804 | 8/10 | 0,713 |

El número débil de la versión es E-extrap 0,786 con 4/8 puros, y se declara como tal. La integración lo degrada: la candidata intacta tenía 0,857 (6/8 puros), a las 15 épocas baja a 0,786 y a las 20 queda en 0,804. Ningún otro indicador retrocede: testD y mono-hist quedan intactos y VE2 se mantiene en 9/10.

## Alcance de la validación

El criterio F1' (testC ≥ 0,90 y E-interp ≥ 0,90 en el punto de 15 épocas) se cumple, con un alcance que conviene declarar con precisión: el criterio se registró después de que el cierre y la revisión de E29 publicaran exactamente las métricas que evalúa, y E29b las replica de forma determinista. La revisión verificó la replicación desde los datos crudos: 15 ficheros eval byte-idénticos, 1.992 filas con 0 discrepancias de id/gold/pred, baselines byte-idénticos, las 15 accuracies por fase reproducidas exacto y el hash de la candidata confirmado byte a byte.

Dicho en claro: es la formalización pre-registrada de un punto ya observado, no una prueba independiente. Confirma el determinismo del pipeline y fija el criterio, pero su desenlace era conocible de antemano. Es además el segundo intento contra la misma barra 0,90/0,90 (E29-F1 falló por una fila: E-interp 0,8966 frente a 0,90), y el curriculum de 15 épocas se diseñó hacia el punto medido que pasa. Esta sección existe para que nadie cite esta versión como validación independiente: no lo es.

## La historia en seis líneas

- E24 (ajuste): 48 problemas con un campo numérico y umbral fijo. FIT_CONFIRMED en 2/2 semillas, pero el umbral queda memorizado: ante umbrales nuevos, las 48 predicciones siguen la regla value ≥ 60.
- E25 (atajo de decena): con umbrales variables, testC 1,000. La revisión posterior encuentra el atajo: la regla misma decena acierta 47/48 de train sin implementar la comparación.
- E27 (la rejilla rompe el techo): con 96 filas el techo era sub-0,90 en 6 semillas (0,625-0,896). La rejilla densa (92 valores × 8 umbrales) lo rompe: testC 0,979/1,000 y sd_n 17/17 y 16/17, antes 8-10/17. Cobertura confirmada.
- E28 (bordes): extremos y cadenas de monotonía. E-extrap sube +16,1/+21,4 pp por semilla y la monotonía llega a 9/10; s43 queda casi completa (testC 1,000, E-interp 0,931, E-extrap 0,857, VE2 9/10).
- E29 (olvido y replay): la candidata intacta tenía la tarea original en 0,157. Con mezcla y replay no hay catástrofe en ninguna dirección: la caída de lo fresco queda en un cuarto a un tercio de la sin replay (E-interp −3,4 pp frente a −12,1; E-extrap −5,4 frente a −16,1), la monotonía pasa de 9→8 con replay frente a 9→0 sin él, y la tarea se recupera a 0,71-0,72.
- E29b (criterio): F1' cumplido por replicación determinista del punto de 15 épocas, con el alcance descrito en la sección anterior.

## Metodología

1. Caps anti-atajo por construcción, con su alcance exacto. Cada valor de la rejilla aparece con 4 pares Sí y 4 No idénticos, de modo que cualquier regla que ignore la pregunta queda exactamente en 0,500 sobre e27-grid (verificado: 0,5000). Las cadenas de e28-ext relajan ese cap exacto: la mejor regla que ignora la pregunta alcanza 0,5424 y el mejor umbral fijo 0,5064 en esa parte del train. Allí el anti-atajo es débil, pero no nulo.
2. OOD incontaminable por construcción. testE usa umbrales de dos decimales, imposibles de colisionar con cualquier umbral de rejilla (un decimal) o histórico.
3. Protocolos congelados antes de lanzar, lecturas pre-registradas por hipótesis y lecciones documentadas cuando la regla no cubría el desenlace. La llamada región del casi-éxito apareció tres veces (E26b D3, E28b W1/W2, E29 F1) y dejó regla formal en docs/decisions.md: una puerta cuyas ramas no cubren el desenlace no puede gobernar lanzamientos.
4. Revisión adversarial en los 9 cierres: 8 por la sesión peer y el cierre de E29b (junto con el informe de versión) por dos agentes independientes de contexto fresco. Las correcciones están integradas y visibles en commits: E24 (66e51957), E25 (08ae3378), E26 (5d1d6f7a, 6 correcciones), E26b (ee68d347), E27 (842353d2), E28b (a26ee010, 3), E29 (5521509e, 2); E28 salió limpia. De los 7 cierres propios con revisión peer, 3 llevaron correcciones. La revisión de E29b corrigió el titular del propio informe de versión, y esa corrección es la que da título a estas notas.
5. Verificador independiente de etiquetas por experimento: 51 o más literales defectuosos cazados antes de lanzar (decenas mal colocadas, colisiones de valores, un par con el miembro Sí por debajo del umbral).

## Cómo reproducir

- Generación y verificación de datos: scripts/gen_e24_fit.py, gen_e25_thrvar.py, gen_e26_order.py, gen_e27_grid.py y gen_e28_ext.py, cada uno con su verificador (verify_e24_fit.py, verify_e25_thrvar.py, verify_e26_order.py, verify_e27_grid.py, verify_e28_ext.py). Informes por experimento: e24_report.py a e29b_report.py (incluidos e26b_report.py, e28b_report.py y e29_report.py). Los datasets quedan en data/e26-order/ (402 filas), data/e27-grid/ (938), data/e28-ext/ (944) y el manifiesto de mezcla en data/e29-mix/.
- Suite de tests: 173/173 en verde al cierre.
- Trazabilidad: runs en runs/kaggle-typed-e24 a e29b, con hash sha256 por artefacto en runs/e*-analysis/artifact_hashes.json. Log completo de la campaña en docs/experiments.md (83 commits al cierre del informe).
- Empaquetado: scripts/export_release.py es el mecanismo de release existente de la línea principal del repo. Verifica los artefactos de entrada (checkpoint, calibration.json, policy.json, run_report), exporta un directorio autocontenido con manifiesto de hashes sha256 y ejecuta una autocomprobación que carga el modelo desde ese directorio y puntúa un caso sin red.

## Límites y estado

- E-extrap 0,786 (4/8 puros) es el punto débil declarado; la extrapolación a valores fuera de rango es lo que la integración degrada.
- La validación del criterio F1' es formalización pre-registrada por replicación determinista, no prueba independiente (sección "Alcance de la validación").
- Estado frente a Julia-1: ventaja no demostrada (−2,1 pp en su benchmark).
- Tras E29b no hay nada automático en marcha. Las opciones abiertas son aceptar la versión, ajustar la mezcla, extender (más campos numéricos, otros operadores, escala) o congelar la línea y documentar.
## Actualización E30 (2026-10-11): validación independiente

Con una muestra original disjunta (congelada antes de conocer el desenlace), la integración cumple el mismo criterio: testC 0,979 · E-interp 0,922 (8/8 puros) · VE2 9/10 · tarea 0,733. La receta de integración no es específica de la muestra. E-extrap 0,750 (2/8) sigue siendo el número débil de la familia.
