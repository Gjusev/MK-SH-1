# Experiments

One row per executed experiment. Artifacts under `runs/` (gitignored; hashes
in `runs/**/run_report.json` and dataset manifests). Nothing here is projected.

## E1 — TF-IDF + logistic regression (local CPU)

- Data: pilot build (`data_hash c4354b6b9bda6b44`), train utterances 54,606
  (286 classes over the three sources).
- Result: banking candidates 98.7 % / banking full-77 86.5 % / massive 96.0 % /
  clinc 98.8 % / clinc_oos 0 % / xprobe 0 % (never picks `none` — structural).
- Latency p50 0.4 ms/caso. Artifacts: `runs/local-baselines/`.
- Rerun on full-budget build (`data_hash a029d9662fc04a99`): banking 98.4 % /
  full-77 87.2 % / massive 96.3 % / clinc 98.9 % / OOS sets 0 %.
  Artifacts: `runs/local-baselines-full/`.

## E2 — Pilot OptionScorer training (Kaggle T4, kernel v5)

- Config pilot.yaml: 7,884 instances (K∈[4,8], 50 % none-distractor, 5.5 %
  none-gold), 2 epochs, eff. batch 32, LR 2e-5/1e-4, fp16, seed 42.
- 494 steps, ~251 s/epoch, peak VRAM 6.63/14.56 GB. Dev_eval (800): acc
  0.9563, NLL 0.164. Checkpoint 1.23 GB (safetensors) recovered + CPU reload
  verified (0.965 on 200 dev cases).
- Kernel: gjusev/mokka-decisions-pilot v5. Artifacts: `runs/kaggle-pilot/`.

## E3 — Pilot calibration + full evaluation (kernel v7)

- Temperature 1.763 (cal_temperature n=4,088; the model is overconfident).
- Threshold 0.300 (cal_policy n=4,280): risk 3.86 % @ 99.86 % coverage.
- Test (see evaluation-protocol.md table): candidates banking/massive/clinc
  0.970/0.954/0.958; OOS clinc 0.937 (TF-IDF 0); xprobe 0.797 with 20.3 %
  false acceptance; full-77 0.705; permutation flips 0/500; ECE 0.011–0.042.
- Kernel v7. Artifacts: `runs/kaggle-eval/` (rows + eval_results.json).

## E4 — Resume test (kernel v8) — PASS

- Continuo 2 épocas: dev acc 0,870 / NLL 0,532. Reanudado (1 época + recarga
  completa de checkpoint + 1 época): 0,860 / 0,438. Δacc 0,010 ≤ tolerancia
  0,05 (no determinismo CUDA). Artifacts: `runs/kaggle-v8/`.
- Transporte del checkpoint verificado vía dataset privado versionado
  `gjusev/mokka-decisions-checkpoint` (mecanismo preferido sobre kernel_sources).

## E5 — Baselines: laya ejecutado; gliner reintento preparado

- Laya multilingüe zero-shot (kernel v8, mismos eval sets, p50 ≈ 39 ms GPU):
  banking candidatos 0,725 · 77-vías 0,390 · massive 0,545 · clinc 0,936 ·
  clinc_oos 0,985 · xprobe 0,815. ECE hasta 0,51 (sin recalibrar).
- GLiNER2.5: carga fallida con transformers 5.19 (AutoExtractor TypeError).
  Parche: instalar `gliner2[local]` + `transformers<5` contenido en la sesión
  del kernel después de construir Laya; reintento tras el run final.

## E6 — Full-budget train + eval (kernel v9) — COMPLETADO

- Receta train.yaml: 16.784 instancias, 3 épocas; mejor época 2; 1.575 pasos;
  dev_eval (1.200) acc 0,9683 / NLL 0,148. data_hash `a029d9662fc04a99`.
- Temperatura 2,349 (cal_temperature n=4.088). Métricas v2 (Brier multiclase).
- Test completo (full budgets): banking candidatos **0,9860** (TF-IDF 0,984 —
  ahora por delante) · massive 0,9687 con de/en/es = 0,9668/0,9681/0,9709
  (equilibrado) · clinc 0,9775 · 77-vías 0,8120 (TF-IDF 0,872 — sigue detrás)
  · clinc_oos 0,9325 (FA 6,75 %) · xprobe 0,7850 (FA 20,75 %) · permutación
  0/500 flips · ECE 0,005–0,150 (máximo en xprobe).
- Selección en test con la política de cal_policy: riesgo 1,4–3,1 % en los
  cuatro escenarios de candidatos (objetivo ≤5 % cumplido ahí), 17,4 % en
  77-vías y 21,5 % en xprobe — fuera del alcance validado, documentado.
- Artifacts: `runs/kaggle-final/` (model_best.safetensors, calibration.json,
  eval_results.json, policy.json, rows/).

## E7 — Comparadores finales sobre IDs idénticos (kernel v10, en ejecución)

- tfidf cerrado + tfidf_lr_none (rechazo entrenado) + laya + gliner (venv
  aislado con transformers<5) sobre los mismos eval sets completos; umbral
  propio por baseline ajustado en cal_policy (riesgo objetivo 5 %).


## E8 — typed-decisions zero-shot (protocolo congelado v1.0)

- Adaptador conforme (json.dumps como julia/data.py, max_length 512, T=1.0,
  batch 1-vs-16 verificado 0 discrepancias, censo completo p99=530 tokens con
  121/7.100 pares truncados, latencia por fila): **0,372 (744/2.000)**
  — choice 0,305 · noul 0,505 · score 0,3225 (MAE esperado 0,944) · NLL 1,38.
- Run previo con settings no conformes (str(), 256, T=2,349): 0,3615 — conservado
  y etiquetado. **Atribución: el adaptador explica ~1 pp; el resto es
  distribución de entrenamiento/dominio** (nunca visto el formato tipado).
- Referencia: Julia reproducida 0,7255 (entrenada en este estilo; presupuestos
  de adaptación distintos, comparación directa no válida).
- Latencia CPU p50 317 ms / p95 366 ms por pregunta (mmBERT-base, K variable).
- Artefactos: `runs/typed-decisions/mokka-base-protocol/`.

## E9 — typed-decisions brazos adaptados (kernel mokka-typed-arms v4) — COMPLETADO

- Mismos datos typed-train (4.950/570/480), adaptador congelado, test pineado:
  - **base_fresh** (307M): dev 0,6912 · **test 0,678** (1.356/2.000) — choice 0,685 · noul 0,758 · score 0,6125 (MAE 0,558).
  - **small_fresh** (140M): dev 0,6386 · **test 0,5345** (1.069/2.000) — choice 0,518 · noul 0,677 · score 0,440 (MAE 0,784).
  - base_init saltado (glob de checkpoint no cubría la ruta anidada del montaje; corregido con rglob).
- **Lectura causal**: zero-shot 0,372 → adaptado 0,678 (+30,6 pp con nuestros datos) — el gap zero-shot era de ENTRENAMIENTO. El gap restante vs Julia reproducida (0,7255): −4,8 pp con Base y **−19,1 pp con igual tamaño (140M vs 144M)** — es de ARQUITECTURA/RECETA (Julia: secuencia única con marcadores + cabezas tipadas; nuestro A0 repite K pares sin tipo).
- Implicación: A1 (marcadores, ya implementada) es el experimento correcto siguiente; mismo datos, receta igual → aísla arquitectura.
- Artefactos: `runs/kaggle-typed-arms-final/` (json por brazo + pesos).

## E10 — typed_a1 (kernel mokka-typed-a1) — COMPLETADO: resultado NEGATIVO

- MarkerScorer (una secuencia por decisión, marcadores [O_i], mmBERT-base) con mismos datos, inicialización y receta que A0 base_fresh.
- Curva dev: 0,5123 / 0,5912 / 0,5930 — claramente por detrás de A0 (0,614/0,625/0,663) en cada época.
- **Test: 0,5345 (1.069/2.000)** — choice 0,498 · noul 0,708 · score 0,431. A0 fue 0,678 → **A1 es −14,4 pp peor**. Cero truncamientos (0 casos perdidos).
- Latencia p50 342 ms/pregunta (GPU, batch 8 amortizado; no comparable directamente con el 317 ms CPU de A0 batch=1 — medición conjunta pendiente si la rama sobreviviera).
- **Interpretación**: la arquitectura de marcadores tal cual NO explica la ventaja de Julia; empeora con nuestra receta. La brecha restante de A0 vs Julia (−4,8 pp) apunta a receta/pérdidas/datos de entrenamiento (H1: pérdida ordinal; después: épocas/LR si hiciera falta), no a «una sola secuencia».
- Hipótesis A1 DESCARTADA con esta configuración; código y pesos conservados (runs/kaggle-typed-a1/). Sesión cancelada al confirmar completado para preservar cuota (total ≈ 5 h wall).

## E11 — typed_ord / H1 (kernel mokka-typed-ord) — CONFIRMADA en dev

- Hipótesis: las confusiones ordinales adyacentes confiadas se corrigen con objetivo unimodal (soft targets, tau=1) en filas score (1.980 de 4.950 train).
- **Dev: 0,7123 vs control 0,6912 (+2,1 pp)** — mejora en los tres tipos: choice 0,7471 · noul 0,7733 · score 0,6404. MAE dev 0,584.
- **Test histórico (exposición declarada, no usado para seleccionar): 0,7045 (1.409/2.000)** vs base_fresh 0,678 → +2,65 pp por la pérdida ordinal. choice 0,698 · noul 0,818 (+6,0) · score 0,624. MAE 0,602. Brecha con Julia reproducida (0,7255): **−2,1 pp** (era −4,8).
- Latencia: 1.448 s / 2.000 preguntas ≈ 724 ms/preg (CPU 8 hilos, batch=16); Julia 324 s ≈ 162 ms/preg (mismo CPU, batch=1). Tiempos totales amortizados con batching distinto — no son p95 comparables; registrado como observación de ejecución, no como veredicto de coste.

## E12 — small_ord / transferencia de receta (kernel mokka-typed-small-ord) — COMPLETADO

- Misma receta ganadora en mmBERT-small (140M, 4 épocas, cabeza A0): dev 0,6439 vs small_fresh 0,6386 (**+0,5 pp** — transferencia débil en esta configuración). Por tipo: choice 0,665 · noul 0,709 · score 0,579.
- Lectura: la receta ordinal ayuda a Base y poco a Small con la cabeza A0; a igual tamaño seguimos lejos de Julia (small_fresh test 0,5345 vs 0,7255). **No concluye límite de capacidad**: Julia usa otra arquitectura sobre mmBERT-small (capas Transformer adicionales, embedding de tipo, marcadores MASK) — replicarla es un brazo separado.

## E13 — Test histórico desglosado por dominio (particiones frozen v1.0)

- **Corrección 2026-10-08 (revisión independiente)**: esto NO es un holdout independiente. Los 4 configs de dominio son una partición de all/test, cuyo agregado ya se había consultado — se registra como desglose del test histórico con exposición declarada. Los números por dominio no se usaron para ajustar nada. Un conjunto independiente nuevo se diseña aparte (ver docs/typed-independent-eval-protocol.md).
- Julia (motor oficial, CPU 8 hilos): agent_trace 348/500=0,696 · customer_service 352/500=0,704 · invoice 398/500=0,796 · security 353/500=0,706 · **TOTAL 1.451/2.000=0,7255** — coincide exactamente con la reproducción agregada (integridad de particiones confirmada). Rerun con dump per-case verificado determinista. Artefactos: `runs/typed-decisions/julia_holdout_domains.json` + `julia_holdout_percase.jsonl`.
- base_ord (protocolo idéntico, per-case dump): agent_trace 336/500=0,672 · customer_service 355/500=0,710 · invoice 382/500=0,764 · security 336/500=0,672 · **TOTAL 1.409/2.000=0,7045** — coincide exactamente con su test histórico. Por dominio vs Julia: −2,4 / +0,6 / −3,2 / −3,4 pp. score MAE 0,497–0,789 por dominio. Artefactos: `runs/typed-decisions/base_ord_holdout_domains.json` + percase.
- **Emparejado por pregunta (McNemar exacto, n=2.000)**: diff +2,1 pp · discordantes 235 solo-Julia vs 193 solo-Mokka · p=0,047 · CI bootstrap por caso original [0,000, 0,042] (revisión: 50.000 remuestreos sobre 400 grupos). La ventaja de Julia sobre base_ord es fronteriza en esta métrica. Artefactos: `runs/typed-decisions/base_ord_holdout_paired_stats.json`.

## E14 — H2 épocas extendidas (kernel mokka-typed-ord v2) — COMPLETADO: sin mejora

- Motivación: curvas dev sin saturar — base_ord +2,8 pp en su última época (0,6842→0,7123), small_ord +3,5 pp (0,6088→0,6439).
- Diseño: **typed_ord2** = base_ord con epochs 3→6 (base_ord6) y **typed_small_ord2** = small_ord con 4→8 (small_ord8). **Confesión de diseño (corrección 2026-10-08)**: cambiar `epochs` cambia `total_steps` y por tanto el calendario de LR (warmup + decaimiento) desde el paso 1 — H2 NO equivale a «3 épocas más de lo mismo»; es una receta distinta desde el arranque.
- **base_ord6 dev 0,7333** (época 4) vs 0,7123 → elegido por dev. **small_ord8 dev 0,6404** vs 0,6439 → sin mejora en dev.
- **Test histórico desglosado (misma partición de E13): base_ord6 0,6925 (1.385/2.000)** — sin mejora sobre 3 épocas (0,7045). Por dominio: agent_trace 0,710 (+1,4 vs Julia) · customer_service 0,616 · invoice 0,762 · security 0,682. McNemar vs Julia: +3,3 pp, p=0,0025. Artefactos: `runs/typed-decisions/base_ord6_holdout_*`.
- **Lecturas (recalificadas por la revisión 2026-10-08)**: (1) divergencia dev→test documentada — con dev de 114 casos/570 preguntas una inversión de ranking es posible; NO se infiere que todo dev de 570 sea inválido. (2) El descenso de customer_service (−9,4 pp vs 3ep) es descriptivo; NO estima varianza run-a-run (todas las comparaciones usan seed=42; no hay réplicas). (3) Bootstrap agrupado por caso (revisión): 3ep − H2 = +1,20 pp, CI [−0,95, +3,35] — H2 no mejora, pero el deterioro frente a 3ep NO es concluyente; Julia − H2 = +3,30 [+1,05, +5,60].
- **H2: sin mejora.** No se afirma que la receta esté agotada ni que Small esté limitado por capacidad (nuestros brazos Small usan la cabeza A0, no el diseño de Julia — ver E12/E16).

## Síntesis de la investigación enfocada (2026-10-08, corregida)

- **Ventaja sobre Julia-1 NO demostrada** — mejor resultado nuestro: base_ord 0,7045 vs 0,7255 (−2,1 pp; McNemar p=0,047 fronterizo, bootstrap por caso [0,000, 0,042]). En coste: 307M vs 144M parámetros. **Corrección de latencia**: los 724 ms/preg (nuestro, batch=16) y 162 ms/preg (Julia, batch=1) son tiempos totales amortizados con batching distinto — NO son p95 comparables ni aísla arquitectura; se registran como tiempos de las ejecuciones observadas. Medición de servicio en condiciones iguales: pendiente.
- **+2,65 pp ordinal (E11)**: mejora observada en un run (seed 42) y en este test, consistente en dev y test — **pendiente de replicación por semillas** (E15 replica control; las réplicas del candidato quedan condicionadas a señal y cuota).
- Negativos medidos sin ambigüedad: A1 marcadores tal cual (−14,4 pp; con el defecto de conteo corregido y sin atribuirle la pérdida), H2 épocas (sin mejora). Small: **no probado** que esté limitado por capacidad — probado con cabeza A0 y 4/8 épocas, no con arquitectura fiel.
- Presupuesto de entrenamiento de Julia: **desconocido** (pipeline privado según su model card). Retirada la afirmación de que entrenó con las 6.000 preguntas sin reservar validación.
- Señal del dataset aún sin explotar: todas las preguntas de train llevan `gold.probabilities` (5.977/6.000 con más de una opción positiva, sumas ≈1) — la pérdida ordinal actual fabrica la distribución y las descarta. → **E15**.
- **Decisión de suite**: Nano/Vision/Fast/Qwen pendientes de evidencia propia. Próximos brazos por señal: E15 (supervisión con probabilidades originales), luego control Small fiel a la arquitectura publicada de Julia (separando arquitectura de supervisión).

## E15 — Supervisión con probabilidades originales del dataset (kernels mokka-typed-e15 v1-v3) — SEÑAL CONSISTENTE REPLICADA

- Hipótesis: las `gold.probabilities` originales del train (5.977/6.000 multi-positivas) supervisan mejor las filas score que el objetivo artificial exp(−|j−gold|/τ).
- Diseño: control (receta base_ord 3ep, objetivo artificial) vs candidato (idéntico salvo targets score = distribución original vía soft-CE); reentrenados en el MISMO kernel para comparabilidad dentro de sesión. Datos/inicialización/batch/pasos/calendario/semilla emparejados. Selección pre-fijada: dev accuracy. Probabilidades preservadas con verificación de alineación por id/orden/suma/soporte (scripts/prepare_typed_probs.py); argmax≡etiqueta en 98,9%.
- **v1 (seed 42)**: candidato 0,6947 vs control 0,6754 (+1,93 pp) — KL/NLL/Brier/MAE todos mejoran.
- **Réplicas emparejadas (seeds 42/43/44, pares dentro de sesión)**: victorias 3/3 — Δacc **+1,93 / +0,35 / +1,75 pp** (media +1,35, rango [+0,35, +1,93]). KL mejora en 3/3 (todos los candidatos 0,267–0,285 por debajo de todos los controles 0,298–0,304); Brier 0,406–0,423 vs 0,447–0,454; MAE 0,561–0,574 vs 0,603–0,608. Seed 45 (exploratoria, no decide): candidato 0,6807 — dentro del rango de controles; coherente con varianza de semilla, no usada para selección.
- CIs agrupados por caso de cada par individual incluyen 0 (n=570/pair) — la decisión descansa en la consistencia 3/3 + separación uniforme de las métricas probabilísticas, según el criterio fijado ANTES de las réplicas.
- Patrón por tipo: la ganancia se concentra en **noul** (+2,3 a +7,0 pp) aunque su entrenamiento es idéntico — los soft targets en score regularizan globalmente; score mixto (−3,5 a +2,6); choice ≈0.
- Incidente v2 y recuperación: el kernel murió en el 5.º brazo (hipótesis documentada: disco de trabajo agotado ~19,7 GB en best+latest; log de sesión no disponible — artefacto vacío). Recuperados control s43/s44 + candidato s43 completos del output de error; v3 re-entrenó probs s44 (completo) + s45 con limpieza de checkpoint_latest por brazo. GPU verificada por VRAM en train_log (pico 10,6–10,7 GB, 460 pasos/brazo). Cross-check kernel↔local EXACTO en los 7 brazos.
- Varianza cross-sesión documentada (hipótesis abierta, sin atribución): base_ord E11 dev 0,7123 vs e15_control 0,6754 — misma receta/semilla, sesiones distintas.
- Artefactos: runs/kaggle-typed-e15{,-seeds}/ (checkpoints + reportes), runs/typed-decisions/e15_replication_report.json.
- **Estado**: contribución propia REPLICADA (supervisión nativa del dataset). El finalista (e15_probs), el control y Julia quedan a la espera de la evaluación en el conjunto independiente congelado (data/independent-v1, sellado; protocolo docs/typed-independent-eval-protocol.md) — sin ejecutar hasta revisión del usuario.

## E16 — Evaluación en el conjunto independiente congelado (indep-v1) — CERRADO

- Fijación ANTES de abrir resultados (runs/typed-decisions/independent_fixation.json): finalista e15_probs_s44 (dev 0,6965, sha bc704d80…), control e15_control_s43 (dev 0,6877, sha 99d87f49…), criterio dev pre-registrado; semilla 45 excluida. Manifiesto verificado sin regenerar (casos f6869c4a…, generador eea5b1ff…, 150 casos, seed 20261008). Julia: motor oficial, pesos sha df853bf7….
- **Resultado global**: finalista 0,3840 · control 0,3467 · Julia **0,4387** (329/750). Todos caen fuertes frente a sus benchmarks — OOD declarado para todos (familías nuevas, etiquetas por regla).
- **Emparejado por caso (150 grupos)**: finalista−control **+3,73 pp, CI95 [−0,13, +7,60]**, McNemar p=0,107 · finalista−Julia **−5,47 pp [−9,07, −1,73]**, p=0,021 · control−Julia −9,20 pp [−13,07, −5,33], p<0,0001.
- **Contexto por receta (3 semillas, nunca selección)**: probs 0,343/0,411/0,384 (media 0,379) vs control 0,321/**0,441**/0,347 (media 0,370) — rangos solapados (Δ de medias +0,93 pp); la semilla s44 del CONTROL iguala a Julia. La varianza de semilla en OOD (6,8–12 pp de rango) domina las diferencias entre recetas.
- Por tipo: Julia gana en **score (0,54 vs ≤0,23 nuestro)** y sensores (0,69); nuestro modelo gana en noul (0,51–0,55 vs 0,45) y el finalista en choice (0,40 vs 0,35). El colapso nuestro en score OOD sugiere que la serialización tipada + type_emb de Julia transfiere mejor las rúbricas ordinales — motivación directa de E17 (arquitectura Julia en Small, en ejecución).
- NLL/Brier/MAE reportados para nuestros brazos (finalista NLL 1,09 · Brier 0,66 · MAE 1,11); Julia solo acierto (filas sin probabilidades comparables). **Eficiencia: sin mediciones comparables — PENDIENTE, nada deducido de parámetros.**

### Conclusiones separadas (lo pedido)

1. **¿La mejora de E15 se mantiene en casos nuevos?** Parcialmente y NO concluyentemente: el par fijado da +3,73 pp con CI que toca cero; las medias por receta +0,93 pp con rangos solapados. La señal dev (3/3, +1,35 pp) no se transfiere con fuerza a familias nuevas — E15 queda como mejora confirmada en distribución, no confirmada (ni refutada) fuera.
2. **¿Ventaja frente a Julia en el alcance evaluado?** NO en este conjunto: Julia supera al finalista −5,47 pp [−9,07, −1,73] y al control. Sin ventaja demostrada; sin medición de eficiencia comparable (pendiente). El eje score/rúbricas ordinales es la brecha concreta a atacar (→ E17).

- Sin ajustes post-hoc: fijación previa escrita, resultados negativos conservados íntegros. Artefactos: runs/independent-v1/ (6 percase + julia + independent_report.json).

## E17 — Arquitectura Julia en Small (kernel mokka-typed-julia-small v2) — entrenamiento completado, interpretación corregida

- Puerto fiel de JuliaDecisionModel (Apache-2.0; runs/julia-research/upstream-source pineado; divergencias declaradas: act_head RL y temperatura omitidos). Paridad verificada: serialización idéntica token a token y estructura igual que el upstream real (tests/test_arch_julia_serialization.py, 2 PASS). Incidente v1: wheel obsoleto del bundle → ModuleNotFoundError al arranque (45 s GPU); corregido reversionando el dataset antes de v2.
- GPU verificada en log (2×T4, cuda true); 465 pasos por brazo (3 épocas completas); mmBERT-small, mismos datos/splits. El control histórico A0-small usó **4 épocas/620 pasos**: no es una comparación con presupuesto ni calendario idénticos.

| dev histórico (E17 afectado por tipos incorrectos) | ordinal artificial | probabilidades nativas |
|---|---|---|
| A0-small (pares K) | small_ord **0,6439** (E12) | — |
| JuliaArch-small (fiel) | julias_ordinal **0,6421** | julias_probs **0,6649** |

- **Corrección de auditoría (2026-10-09):** el mapa de tipos enviado al evaluador contenía únicamente IDs de train. Ninguno de los 570 IDs de dev estaba presente; el fallback los serializaba todos como `choice`. Esto cambiaba el prefijo y el embedding de tipo de **400 preguntas** (172 `noul`, 228 `score`). Las cifras anteriores se conservan como históricas, pero la selección de checkpoint quedó afectada.
- La semilla se fijaba dentro del trainer, **después** de construir la cabeza aleatoria. No quedó garantizada una inicialización emparejada entre los brazos. Los **+2,28 pp** anteriores no aíslan el efecto de la supervisión y no constituyen una nueva réplica controlada de E15.
- Falta A0-small con probabilidades nativas y el control ordinal tiene otro presupuesto: **no es un 2×2 completo**. Retirada la conclusión «efecto arquitectura nulo»; una diferencia pequeña de un run tampoco demostraría equivalencia.
- Corregido para ejecuciones futuras: mapa explícito train+dev, error ante tipos ausentes/desconocidos, semilla antes de construir el modelo. Nuevas pruebas de regresión incluyen paridad numérica de logits con la cabeza upstream usando pesos compartidos; esta comprobación no replica su entrenamiento privado.
- Se autoriza y ejecuta la evaluación local de los dos checkpoints guardados con tipos correctos. `independent-v1` ya fue consultado en E16: esta evaluación es **exploratoria**. No puede recuperar qué época habría elegido una evaluación dev correcta durante el entrenamiento ni aislar arquitectura/supervisión con los brazos históricos.
- Artefactos: runs/kaggle-typed-julia-small/ (checkpoints_best + run_reports de ambos brazos).

### E17 — Reevaluación de checkpoints guardados con tipos correctos (2026-10-09)

- Ejecución completada en CPU offline, float32, 8 hilos, batch 16, T=1, contexto 1024 y cabecera 256. Plan con hashes congelado antes de inferencia; datasets/generador/checkpoints originales sin cambios. Las 570 preguntas dev y las 750 de independent-v1 pasan la serialización upstream estricta sin truncamiento. Seis pruebas pertinentes pasan, incluida paridad numérica de logits a pesos compartidos.
- Dev correcto de los pesos guardados: ordinal **367/570 = 0,64386**; probs **379/570 = 0,66491**. El cambio agregado frente a las métricas históricas es pequeño; no explica la brecha con Julia ni recupera retrospectivamente la selección de época. El hardware/precisión de esta reevaluación también difiere de la ejecución GPU histórica.

| independent-v1 (exploratorio) | total, n=750 | choice, n=250 | noul, n=300 | score, n=200 |
|---|---:|---:|---:|---:|
| Julia oficial, referencia E16 | 0,43867 | 0,348 | 0,44667 | 0,540 |
| E15 Base probs s44, referencia E16 | 0,38400 | 0,396 | 0,55333 | 0,115 |
| JuliaArch-small ordinal | 0,32533 | 0,248 | 0,51000 | 0,145 |
| JuliaArch-small probs | 0,35867 | 0,312 | 0,37667 | 0,390 |

- Bootstrap pareado por caso, 50.000 remuestreos: Small probs−Julia **−8,00 pp [−12,00, −4,00]**; Small ordinal−Julia **−11,33 pp [−15,07, −7,47]**; Small probs−Small ordinal **+3,33 pp [−0,53, +7,20]**. Intervalos condicionados a estos checkpoints y estas familias; no estiman incertidumbre entre semillas ni generalización a cualquier dominio.
- El checkpoint Small probs recupera acierto score frente a los otros checkpoints propios, pero pierde en otros tipos. No hay ventaja global frente a Julia, ni evidencia causal que permita atribuir el patrón solo a arquitectura o supervisión. E17 sigue siendo una comparación incompleta y con inicializaciones no acreditadas como emparejadas.
- Resultados/hashes/filas: `runs/e17-independent-exploratory-2026-10-09/`; reproducción: `scripts/eval_e17_saved_checkpoints.py --out <carpeta-nueva>`. No se lanzaron nuevos entrenamientos. Para un contraste controlado se necesitan correcciones empaquetadas, presupuesto y orden de datos emparejados, pesos iniciales registrados y réplicas; una nueva confirmación requiere familias selladas.

## E18 — Contraste de supervisión controlado dentro de JuliaArch-small (kernel mokka-typed-e18 v1) — COMPLETADO

- Protocolo congelado antes de lanzar (3 semillas emparejadas 42/43/44 × {ordinal artificial, probabilidades nativas} en filas score; idénticos pesos iniciales por par —verificados por hash del state_dict, abort si difieren—, orden de casos, datos, batch efectivo 32, 465 pasos, optimizador y calendario; selección solo con dev correcto). Fuentes con mapa de tipos explícito train+dev; wheel recompilado y verificado con instalación aislada + smoke test de entrenamiento/evaluación.
- GPU verificada en log: 2× Tesla T4, cuda true, 14,56 GB; hashes de par idénticos en las 3 semillas; cross-check kernel↔local exacto en los 6 brazos.

| dev (seed 42/43/44) | accuracy | NLL | Brier | KL | MAE ordinal |
|---|---|---|---|---|---|
| ordinal artificial | 0,6421 / 0,6825 / 0,6789 | 0,859 / 0,787 / 0,832 | 0,495 / 0,454 / 0,477 | 0,272 / 0,253 / 0,264 | 0,708 / 0,607 / 0,600 |
| probabilidades nativas | 0,6719 / 0,6825 / 0,6404 | 0,772 / 0,753 / 0,853 | 0,447 / 0,434 / 0,491 | 0,244 / 0,233 / 0,280 | 0,566 / 0,549 / 0,688 |

- Δacc por par (probs−ordinal): **+2,98 pp / 0,00 / −3,86 pp**; media **−0,29 pp**, sd entre semillas 3,43 pp; bootstrap agrupado por caso original (114 grupos, 50k): CI95 **[−2,69, +2,22]** — incertidumbre por caso condicionada a estos entrenamientos, reportada aparte de la dispersión entre semillas.
- ΔKL: −0,028 / −0,020 / +0,017 (media −0,010); ΔNLL y ΔBrier mejores en 2/3; ΔMAE mejor en 2/3. La semilla 44 invierte todas las métricas.
- **Veredicto por regla congelada: mejora** (≥2/3 pares sin perder accuracy + KL mejor en media y 2/3 + ninguna métrica empeora uniformemente). Lectura honesta: la mejora la sostienen las métricas probabilísticas (calibración frente a las distribuciones originales); **la ganancia de accuracy NO queda establecida** (media negativa, CI que toca cero a ambos lados, dispersión entre semillas que domina el efecto). A diferencia de E15 (3/3 en accuracy con separación uniforme, cabeza A0-base), dentro de JuliaArch-small el efecto de la supervisión en accuracy es inconsistente entre semillas.
- Alcance declarado: contraste de supervisión dentro de JuliaArch-small; no prueba efecto de arquitectura ni completa el 2×2. Independent-v1 no se consultó en este experimento. Comparaciones de semilla única (como los números dev históricos de E17) siguen sin ser fiables: los controles varían 4,0 pp entre semillas con receta idéntica.
- Artefactos: `runs/kaggle-typed-e18/` (6 checkpoints best + dev_rows por brazo + e18_summary/e18_report + hashes); análisis: `scripts/e18_supervision_report.py`. Manifiesto pre-lanzamiento con hashes de fuentes/wheel/runner/datos: `runs/e18-prep/prep_manifest.json`.
- Siguiente experimento recomendado (según protocolo y pendiente de decisión): transferencia de reglas ordinales con datos verificables nuevos en familias nuevas selladas (nunca independent-v1 ni su generador), para atacar la brecha OOD score identificada en E16 en lugar de seguir ajustando la supervisión en distribución.

### E18 — Fe de erratas (2026-10-09, sin cambio de resultados)

- La KL reportada en E15/E17/E18 es **KL(original ∥ pred)** (Σ v·log(v/p), v=distribución original del dataset, p=predicción); el protocolo interno la abreviaba en sentido inverso. Todos los valores históricos se conservan; no se recalculan.
- La rama negativa del analizador se corrige para coincidir con el criterio escrito (KL empeora en ≥2/3 pares, no 3/3); sin efecto sobre el veredicto E18 (entra por la rama de mejora). En accuracy hubo 1 victoria, 1 empate y 1 derrota — la etiqueta «mejora» no implica mejora de acierto.

## E19 — Datos verificables con variación de reglas y contrafactuales (kernel mokka-typed-e19 v2) — COMPLETADO

- Protocolo congelado antes de generar datos o entrenar (3 brazos × semillas emparejadas, reducidas a 42/43 por regla de cuota pre-fijada; presupuestos igualados 620/621 pasos; convención declarada: probabilidades nativas solo en filas score ORIGINALES, filas nuevas deterministas con CE dura). Datos: 4 familias nuevas disjuntas (insurance_claims_triage, facility_maintenance, it_access_review + **records_retention reservada solo para test**), 330 casos train compartidos entre brazos (ctrl = renderizados base; cand = variación de reglas + pares contrafactuales), 66 dev, 150 test sellado (750 preguntas, hashes congelados antes de entrenar). **Etiquetas 4.380/4.380 verificadas por un verificador independiente** que re-deriva cada oro desde los textos (el proceso detectó y corrigió 6 defectos reales del generador antes de lanzar). Antecedente: Kaushik et al., ICLR 2020 (contrafactuales; no invención nuestra).
- Ejecución verificada: GPU 2×T4 en log, hashes iniciales idénticos entre los 3 brazos de cada semilla, pasos 620/621, ~56 min de pared. Selección solo con dev (900 preguntas = 570 originales + 330 nuevas). Test sellado abierto una vez tras la fijación.

| test sellado (s42/s43) | acc | choice | noul | score | NLL | MAE |
|---|---|---|---|---|---|---|
| orig (sin datos nuevos) | 0,3027/0,2747 | 0,20/0,18 | 0,65/0,47 | 0,23/0,27 | 1,45/1,42 | 0,97/0,97 |
| ctrl (+1.650 ordinarios) | 0,3653/0,3987 | 0,23/0,29 | 0,46/0,47 | 0,46/0,47 | 1,61/1,62 | 0,75/0,68 |
| cand (+1.650 contrafactuales) | **0,4093/0,4147** | 0,34/0,36 | 0,47/0,47 | 0,45/0,44 | 1,14/1,18 | 0,69/0,71 |
| Julia oficial | 0,2853 | 0,23 | 0,42 | 0,27 | 3,92 | 1,40 |

- **Veredicto por regla congelada: MEJORA, confirmado tras corregir el análisis.** Contraste principal cand−ctrl: +4,40/+1,60 pp, media **+3,00 pp**, IC95 por caso **[+1,00, +5,00]**, condicional a estos dos pares. Ambos pares favorables y sin regresión conjunta choice+noul: mejora **+6,22 pp** en ese subconjunto, no +16 pp. Por tipo: choice **+9,17 pp**, noul **+0,33 pp**, score **−1,83 pp**. (i) Añadir los datos ordinarios mejora el acierto +6,27/+12,40 pp frente a orig, con NLL peor. (ii) El paquete candidato añade +3 pp frente al control; los pares action-flip pasan de 1,33/0,67 % a 12,67/18,00 % ambos-correctos. (iii) **Ventaja agregada frente a Julia en este test: +12,67 pp**, IC95 por caso **[+8,40, +17,00]**, con candidato entrenado en tres familias de este generador y Julia oficial sin esa adaptación.
- **Corrección del informe (2026-10-09):** el bootstrap v1 usaba `dict.update` sobre los mismos IDs de caso y sobrescribía la primera semilla; v2 promedia diferencias emparejadas de todas las semillas dentro de cada caso antes de remuestrear. El denominador choice+noul era 170 en vez de 450. Accuracy y predicciones no cambian; reporte original conservado, nuevo en `runs/e19-review-2026-10-09/e19_report_v2.json`. Siete pruebas de regresión pasan. Los intervalos no incluyen variabilidad entre entrenamientos o familias nuevas.
- **Alcance del tratamiento:** cand cambia también la distribución de tareas: reemplaza outcome por otro action y modifica risk/urgency; el test utiliza action/flip y risk/flip. El contraste mide el paquete de datos, no aísla la eficacia del emparejamiento contrafactual frente a mayor cobertura de las tareas evaluadas. `irrelevant_correct` es acierto con ruido añadido, no invariancia pareada: falta la predicción del mismo booleano sin ruido.
- **Límite declarado (familia reservada)**: en records_retention las medias son Julia **19,33 %**, orig **16,50 %**, ctrl **12,00 %**, cand **9,33 %**. En las otras tres familias: Julia 34,67 %, orig 37,11 %, ctrl 55,67 %, cand 62,44 %. La transferencia a la familia nueva empeora respecto a orig; no hay superioridad general demostrada. Sus 60 preguntas noul tienen todas oro false, y ambos candidatos responden true en todas: diagnóstico de sesgo de respuesta en ese conjunto, no prueba por sí solo de su causa. No entrenar sobre este subset y volver a llamarlo test independiente.
- Artefactos: `runs/kaggle-typed-e19/` (6 checkpoints + dev_rows 900/brazo + summary), `runs/e19-sealed-eval/` (7 percase con probabilidades + e19_report + hashes), `runs/e19-prep/` (freeze del test + manifiesto pre-lanzamiento). Generador/verificador/evaluación: `scripts/gen_e19_rulevar.py`, `scripts/verify_e19_labels.py`, `scripts/eval_e19_sealed.py`, `scripts/e19_rulevar_report.py`; tests: `tests/test_e19_generator.py`.

## E20 — Diagnóstico de transferencia + contraste composición/cobertura (kernel mokka-typed-e20 v1) — COMPLETADO

- **FASE 1 (diagnóstico, CPU)**: recorrido noul verificado sin defecto detectado de adaptador; etiquetas re-verificadas. Batería de 480 sondas noul balanceadas (240/240, reglas explícitas de dos condiciones): **los 6 checkpoints E19 y Julia obtienen 50 %** (nuestros predicen «true» en el 100 %, Julia 88 %). Esto acredita un sesgo de respuesta en estas sondas. Los priors por familia son una hipótesis explicativa (train noul: insurance 81 % true, facility 11 %), no una causa aislada experimentalmente ni una prueba de incapacidad general para componer reglas.
- **FASE 2/3 (congeladas antes de entrenar)**: ctrl (+1.650 base, 330 casos) vs cand (+1.650 pares base+flip en los 5 slots, 165 casos), misma mezcla 40/20/40, 621 pasos totales por brazo durante 3 épocas, oros noul 50/50 por familia, semillas 42/43/44 con hashes iniciales verificados 3/3. Test sellado: 2 familias NUEVAS (loan_underwriting, chemical_batch_release), 750 preguntas, hashes congelados; etiquetas 4.380/4.380 verificadas por texto independiente. El test contiene 150 choice, 300 noul y 300 score; su mezcla difiere del train, aunque es idéntica para todos los modelos evaluados.

| test sellado (s42/43/44) | macro reservadas | choice | noul | score |
|---|---|---|---|---|
| ctrl | 0,376/0,413/0,409 | 0,20/0,31/0,20 | 0,45/0,50/0,50 | 0,39/0,38/0,42 |
| cand | 0,432/0,463/0,341 | 0,14/0,09/0,19 | **0,59/0,70/0,34** | 0,42/0,41/0,42 |
| Julia | 0,301 | 0,34 | 0,51 | 0,07 |

- **Veredicto por regla congelada: INCONCLUSO.** cand−ctrl macro: +5,6/+4,9/−6,8 pp, media +1,24 pp (< +5 exigido), CI por caso [+0,04, +2,44]; la semilla 44 invierte. Este intervalo está condicionado a los tres pares observados y no incluye variabilidad entre entrenamientos o familias. El tratamiento mejora **noul** en 2/3 semillas, pero choice cae −6,00/−22,00/−1,33 pp (media **−9,78 pp**). En el dev nuevo de familias conocidas cae −9,39/−9,09/−15,76 pp (media **−11,41 pp**). Ambas regresiones exceden el límite congelado de 2 pp. El contraste también cambia cobertura: el candidato ve la mitad de casos distintos, por lo que no aísla el emparejamiento de su coste en diversidad. La batería diagnóstica posterior da ctrl/cand 64,17/65,83 % en s42, 50,00/51,88 % en s43 y 50,00/50,00 % en s44.
- **Frente a Julia**: ambos brazos superan a Julia en las familias selladas nuevas (cand−Julia **+11,1 pp [+8,1, +13,9]**; Julia colapsa en score 0,073, NLL 4,13). Misma salvedad de procedencia: familias de nuestro generador; Julia sin adaptación a familias vistas por nuestros brazos.
- Artefactos: `runs/kaggle-typed-e20/` (6 ckpts + dev_rows), `runs/e20-sealed-eval/` (7 percase + report + hashes), `runs/e20-diag/` y `runs/e20-diag-post/` (batería pre/post), `runs/e20-prep/` (freeze + manifiesto). Scripts: gen_e20_diag/eval_e20_diag/gen_e20_contrast/verify_e20_labels/eval_e20_sealed/e20_report; tests 65/65.
- **Correcciones de revisión (2026-10-09):** `e20_report.py` tomaba la mejor diferencia de cualquier tipo/semilla para comprobar regresiones. La versión 2 comprueba la diferencia media de cada tipo y el dev conocido por separado, exige los checks conocidos completos antes de declarar mejora y explicita que falta evaluar los checkpoints E20 en las familias conocidas del diagnóstico E19. Se añaden guardas de alineación, dispersión entre semillas y nueve pruebas de regresión. Veredicto y métricas históricas no cambian; informe v1 preservado y nuevo en `runs/e20-review-2026-10-09/e20_report_v2.json`.
- **Límites adicionales del diagnóstico:** las 96 variantes de paráfrasis son copias exactas del original (dos caracteres backspace en el regex del generador); ninguno de los 96 cambios de umbral altera el oro, por lo que no prueba si el modelo cambia de respuesta al cambiar la regla. En 47/96 factflips cambian dos hechos, no uno. La revisión re-deriva las 480 etiquetas sin discrepancias, pero estos defectos limitan la interpretación causal. Datos, generador y manifiesto históricos se conservan sin modificaciones; una batería corregida debe llevar otra versión y seguir siendo exploratoria.
- Conclusión: esta receta no satisface los criterios de mejora; queda inconclusa la hipótesis de transferencia. No demuestra que la composición sea imposible de aprender, ni que los pares contrafactuales fallen en general. Eficiencia comparable frente a Julia sigue pendiente.

## E21 — Diagnóstico v2 por habilidades + medición exploratoria de inferencia (CPU, sin entrenar) — ejecución completada, comparabilidad pendiente

- Integradas las correcciones de la revisión E20 (analizador con medias por tipo, dev de familias conocidas, empates; commit 66d0bc5). E20 permanece INCONCLUSO.
- **Diagnóstico v2** (`data/e20-diag2/`, 312 filas noul): en las 96 filas de comparación numérica simple, los seis checkpoints E20 y Julia obtienen 0,49–0,50 de accuracy. El fallo aparece sin exigir composición en este formato, pero no distingue extracción numérica, comparación, formato JSON o sesgo sí/no. Los pares de inversión dan 0,000 ambos-correctos, salvo cand_s44 con 0,25. Los tests verifican cambios reales de paráfrasis, inversiones y ruido. El generador histórico obtiene directamente el oro de su checker y reutiliza `num_holds`: no constituía una verificación independiente. La revisión posterior re-derivó las 312 etiquetas con otro parser y operadores, sin discrepancias, y comprobó alineación/aciertos de las siete evaluaciones guardadas.
- **Medición exploratoria de inferencia** (CPU, 8 hilos, float32, mismas 750 preguntas, max_length 512; calentamiento + 3 pasadas × 2 invocaciones válidas por modelo; predicciones verificadas idénticas a la evaluación oficial): Mokka usa lotes de 16 y Julia llamadas individuales. Las cifras históricas siguientes miden unidades distintas y no constituyen una comparación homogénea de latencia.

| modelo | batch real | unidad del percentil | p50 | p95 | throughput observado | pesos |
|---|---|---|---|---|---|---|
| Mokka ctrl_s43 (mejor dev) | 16 | duración del lote / 16 | 67,8 ms | 71,3 ms | 15,9 preg/s | 550 MB |
| Mokka cand_s43 | 16 | duración del lote / 16 | 67,7 ms | 69,7 ms | 16,0 preg/s | 550 MB |
| Julia oficial | 1 | latencia de petición individual | 100,5 ms | 135,1 ms | 9,8 preg/s | 550 MB |

- **Corrección de comparabilidad:** retirado el claim «1,5× más rápido». El p50 del lote Mokka completo es aproximadamente 1,084 s; dividirlo por 16 mide coste amortizado y no latencia individual. Esto tampoco demuestra que Mokka sea más lento a igual batch: esa comparación sigue pendiente. El motor Julia sí admite lotes con estados distintos mediante `TransformerEngine.logits(rows)`; la restricción procede de cómo se llamó a `predict_typed`. El campo histórico `batch_questions=16` también se escribió para Julia pese a usar batch 1, y el último lote Mokka de 14 se dividió por 16. El throughput corresponde a los intervalos internos cronometrados, no a un recorrido completo con todo el procesamiento de salida. Memoria pico no medida; tamaño de pesos no equivale a memoria en ejecución. Selección por dev preservada (ctrl_s43 0,7378; cand_s43 0,7022).
- **Alcance del diagnóstico:** 312 filas, 184 contenidos de entrada distintos; numcmp tiene 92 contenidos distintos, catcond 12/24 y el bloque de pares 48/160. Hay 184 IDs y 312 claves únicas `(id, variant)`; la evaluación actual procesa todas las filas, pero indexar solo por ID perdería variantes. Catcond tiene 8 true/16 false (baseline mayoritario 2/3); AND y OR tienen baseline mayoritario 3/4. No se deben interpretar sus resultados contra un baseline uniforme de 1/2 ni las repeticiones como observaciones independientes. La generación usa la misma plantilla de pregunta para todos los casos.
- **Hipótesis siguiente, todavía no confirmada:** entrenamiento con condiciones numéricas simples y pares de inversión. Para atribuir una mejora al currículo, comparar el orden por fases frente a los mismos ejemplos mezclados, con presupuestos e inicializaciones emparejados y evaluación nueva. Mejorar frente a un checkpoint sin entrenar no aísla el efecto del orden. Superar 0 en pares tampoco basta (respuestas binarias independientes uniformes alcanzan 0,25 en promedio). Ni un éxito demuestra una causa única ni un fracaso demuestra un límite arquitectónico.
- Revisión local: `runs/e21-review-2026-10-09/verification.json`. Datos, fuentes y resultados históricos conservados. Las siete pruebas existentes del diagnóstico pasan, pero no detectaban los límites anteriores. Pendiente medir batch 1 y 16 para ambos modelos con intervalos cronometrados equivalentes; no se ha ejecutado esa medición en esta revisión.
- Artefactos: `data/e20-diag2/`, `runs/e20-diag2/`, `runs/e21-efficiency/` (raw + summary con salvedades); scripts: `gen_e20_diag2.py`, `eval_e20_diag2.py`, `efficiency_e21.py`; tests 81/81.
