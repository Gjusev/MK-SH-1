# Experiments

One row per executed experiment. Artifacts under `runs/` (gitignored; hashes
in `runs/**/run_report.json` and dataset manifests). Nothing here is projected.

## E26 — ORDEN numérico con atajo de decena ROTO + control de presupuesto igual (kernel mokka-typed-e26 v1) — COMPLETADO

- Dos brazos con filas/pasos/presupuesto idénticos (96 filas, 900 pasos efectivos máx): **tratamiento** con 10 umbrales NO alineados con decenas (41-92) y cobertura completa de cuadrantes — negativos misma-decena y positivos lejanos, imposibles con umbrales alineados — más pares solo-umbral Y solo-valor; **control** con 8 umbrales alineados (30-100) estilo E25. Guardas sobre datos: regla misma-decena 0,604 en train del tratamiento (atajo roto) vs 0,917 en control. Umbrales reservados jamás inspeccionados (interp 44,5-88,5, extrap 25/115); testD con pares PUROS interp y extrap; 6 sondas de monotonía. 402/402 etiquetas re-derivadas; el verificador cazó 21 literales mal colocados antes del lanzamiento.
- **V1 AJUSTE: 4/4 brazos** (train 1,000; épocas 59/42 tratamiento, 25/25 control; omisiones AMP registradas 3-4 por brazo; hardware 2× T4 verificada en hardware.json).
- **V2-V5 (tratamiento): TODOS FALLAN.** V2 testC 0,625/0,875 (<0,90; pares solo-umbral 0,5/1,0, solo-valor 0/0,5). V3 interp 0,553/0,868, extrap 0,438/0,500 — ni cerca con ambas semillas. V4 pares C+D 0,2/0,7. V5 monotonía 0/6 y 1/6 barridos — ni un barrido monótono completo en s42.
- **V6 control: también falla todo** (testC 0,667/0,563; interp 0,737/0,684; extrap 0,562/0,438; pares 0,2/0,2; monotonía 2/6 y 0/6). Rama pre-registrada «ninguno pasa» disparada; el cierre la etiquetó «límite de capacidad/duración» — **la revisión la corrige**: la duración NO fue limitante (parada temprana a 8-19 % del presupuesto: 174/122/72/71 pasos efectivos de 900; la regla de parada, no el presupuesto, es la palanca) y «capacidad» carece de soporte; el fallo del control está sobredeterminado por su propio diseño compatible con el atajo (misma-decena 0,917 en su train, sd_n inexistente por construcción). Mejor respaldado: dificultad del eval + señal de los datos (96 filas) + varianza alta de semilla.
- **Firma por cuadrante (descriptivo, consistente en los 4 brazos):** frontera exacta 7/7 en todas partes · misma-decena-Sí 16-17/17 · **misma-decena-No 0/17 en ambos controles** vs 8-10/17 en tratamiento · cruzada-No 13-22/24 · cruzada-Sí (positivos lejanos) 3-12/17. El heurístico «misma decena→Sí» (48/82 filas base) domina en los controles (57 y 50/82) y en s42-tratamiento (47/82, por debajo de la regla); s43-tratamiento lo supera con claridad (66/82). La resistencia del tratamiento es consistente SOLO en misma-decena-No; en cruzada-Sí el efecto cambia de signo con la semilla (s42: 3/17 del tratamiento vs 11/17 del control — peor) y el tratamiento paga un coste consistente en cruzada-No (13-21/24 vs 22/24 del control). Dispersión de semilla dominante: 25 pp de diferencia en testC.
- Ajustar el train (1,000) NO produce función de orden: la monotonía colapsa en todos los brazos y semillas (0-2/6 barridos; implementación estricta: no-decreciente Y con al menos un No y un Sí — el barrido degenerado todo-Sí de t=115 en ambos controles cuenta como NO monótono; lectura laxa daría 3/6 y 1/6, sin impacto en veredictos). El patrón de colapso del CONTROL es exactamente la banda del heurístico de decena (cruce al entrar en la decena del umbral, re-caída confiada p 0,998→0,011 al salir; 8/8 caídas a ±1 posición de la salida de decena); el del TRATAMIENTO NO lo es (cruza en la posición verdadera en los 6 barridos de umbrales de train; solo 4/10-5/8 caídas coinciden con la salida de decena; oscilación intra-decena que la regla no genera). t=115 degenera a Sí-constante en ambos controles, no predicho por la regla. No es comparación vs Julia ni medida de olvido.
- **Revisión posterior (2026-10-10), cross-eval E25→E26 (CPU, harness validado 48/48):** los finalistas E25 —1,000 en su eval de umbrales conocidos con 47/40 pasos— caen a 0,521/0,542 en testC E26 y 0,537/0,519 en testD, con firma pura de atajo (sd_n 0/10 y 0/7; sd_y 10/10; cd_y 2-3/10; frontera 7/7). El 1,000 de E25 medía el atajo de decena, no competencia de comparación: la caída E25→E26 mide dificultad del eval, no una regresión del setup.
- **Revisión posterior, tasa base (sd_n):** el contraste misma-decena-No tratamiento/control NO es artefacto de tasa de predicción: sobrevive a las tres formas de ajuste (tasa global de No, tasa por cuadrante, bloque misma-decena balanceado 22/22). Es colocación selectiva del No condicional al oro: P(No|misma-dec, oro-No)=9/23 y 14/23 en tratamiento vs P(No|misma-dec, oro-Sí)=1/22 (Fisher 0,006 y 0,00005); los controles predicen Sí constante dentro de la decena (Fisher p=1,0). PERO el coste global es asimétrico: accuracy balanceada s42-trat 0,569 < s42-ctrl 0,676 (el No se derrama a cruzada-Sí); solo s43-trat supera a su control (0,814 vs 0,588).
- **Revisión posterior, criterios:** V2 aplicó la cláusula de pares POR TIPO (solo-umbral ≥0,75 Y solo-valor ≥0,75) cuando el protocolo decía agregada — endurecimiento a posteriori declarado; s43 agregado (2+1)/4=0,75 pasaría la cláusula de pares (V2 sigue fallando por accuracy 0,875<0,90). Ningún veredicto cambia bajo lectura either-seed, agregada o laxa de monotonía. v6_reading añadió una 4ª rama no pre-registrada (no disparó). Referencias que el cierre omitía: en testC los 4 modelos superan la regla misma-decena (0,500); en testD (regla 0,611) solo s43-trat (0,759) y s42-ctrl (0,685) la superan, s43-ctrl empata y s42-trat queda por debajo. `grad_nonfinite_epochs` cuenta eventos por época (puede valer 2), no épocas.
- **Siguiente intervención recomendada (no ejecutada):** E27 «rejilla densa de comparación» — augmentación continua en AMBOS ejes (por cada valor de train, K=6-8 umbrales en rejilla fina valor±{0,5/3/7/15/25}, ~600-750 filas, excluyendo los 7 reservados y los valores de dev/test/mono), parada por dev (paciencia ~10 épocas) en vez de por memorización, modelo/presupuesto/evals E26 intactos, 2 semillas (+2 control opcional). Salidas pre-registradas que separan: cobertura (sd_n≥15/17 e interp≥0,90) vs déficit de representación del operador (train 1,0 pero sd_n≤10/17) vs déficit de orden en valor (celdas altas pero mono<0,80). Discriminadores baratos primero: continuar s43-trat desde final con parada desactivada; 2-4 semillas extra. El estudio de olvido NO discrimina ninguna hipótesis de E26 (los finales ctrl son bit-idénticos a e025 y ese endpoint ya está medido).
- Artefactos: `runs/kaggle-typed-e26/` (4 brazos × {inicial/e025/final, curvas instrumentadas, predicciones, evals known/new/mono por fila}, hardware.json), `runs/e26-analysis/` (reporte con criterios congelados + 38 hashes), `runs/e26-prep/`. Commits: `08ae3378` (revisión E25), `e175b757` (prep), cierre. Olvido sigue PREPARADO, NO lanzado.

## E25 — Comparación con UMBRAL VARIABLE (kernel mokka-typed-e25 v1) — COMPLETADO

- Objetivo: ¿aprende a comparar el valor del estado con el UMBRAL DE LA PREGUNTA (dos operandos variables)? Seis umbrales de entrenamiento (40-90); **pares solo-umbral** con estado byte-idéntico donde SOLO cambia el umbral de la pregunta y el oro se invierte — cualquier regla que ignore la pregunta queda matemáticamente en ≤0,50 de accuracy y 0 pares (verificado sobre los datos: constantes y regla fija `value ≥ 60` dan exactamente 0,50/0,50 en los pares). Umbrales reservados nunca vistos: interpolación 45-85, extrapolación 30/100. Identificadores puramente alfabéticos: el único número del estado es el valor medido. Protocolo congelado antes de generar.
- Datos 48/36/36/54 (train/dev/testC/testD), balance global exacto; por umbral exacto en train/dev/C y diferencia de hasta una etiqueta en D. Valores disjuntos entre los cuatro splits (dos colisiones cazadas por las guardas antes del lanzamiento), 174/174 etiquetas re-derivadas por verificador independiente, 25 tests nuevos con literales manuales (suite 144/144 al cierre).
- Instrumentación corregida por la revisión E24: **hardware.json real (2× Tesla T4, 14,56 GB, verificado)**; intentos vs **actualizaciones efectivas vs omitidas** separados; escala AMP y gradientes no finitos por época marcados (nunca `inf` en JSON); el scheduler avanza SOLO con actualizaciones efectivas. Cuota reconciliada: 329,4 s acumuladas tras E25.
- **V1 AJUSTE: sí (2/2).** Train 1,000 y dev 1,000 ambas semillas; primer perfecto en índices de época 21/17 (épocas 22/18 desde uno), parada tras 24/20 épocas. Contadores registrados: 47/48 pasos efectivos/intentos, 1 omitido por gradiente no finito en s42; 40/40 en s43. E24 y E25 no son una ablación que cambie solo los umbrales: también cambian formato de distractores, presupuesto y calendario.
- **V2 VALORES NUEVOS, UMBRALES CONOCIDOS (testC): criterio cumplido.** 36/36 aciertos y 10/10 pares solo-umbral ambos-correctos por semilla. El comportamiento depende del umbral de la pregunta en estos pares. El bootstrap devuelve [1,00, 1,00] porque todos los casos remuestreados son correctos: es un intervalo degenerado, no certeza de accuracy poblacional 100 %. Esta prueba no excluye atajos que sí usen el umbral.
- **V3 UMBRALES NUEVOS:** interpolación (40 filas) 0,575/0,625, por debajo de 0,90; pares puros-interp 3/4 y 4/4. Extrapolación (14 filas) 14/14 ambas semillas, resultado descriptivo. El criterio congelado exigía también pares puros y no hay ninguno: **veredicto V3b no evaluable**, sin eliminar ese requisito a posteriori. D contiene cuatro pares puros-interp y dos mixtos; los mixtos dan **1/2 en ambas semillas** (el informe original guardaba erróneamente 0 al usar el contador de incompletos). Los errores no se limitan a distancia ≤7,2: s42 también falla a -12,0 y ambas a +7,4.
- **V4 PARES SOLO-UMBRAL C+D: 0,875/0,9375** — el criterio de ≥0,90 en ambas semillas no se cumple (solo s43 supera 0,90). Ambas fallan el par mixto 92.4; s42 falla además un par puro-interp.
- Referencias que ignoran la pregunta (congeladas): en pares 0,50/0,50 — el diseño cumple su función; el 1,000 de testC los supera con margen.
- **Revisión posterior, referencia exploratoria «misma decena»:** `floor(value/10) == floor(threshold/10)` acierta 47/48 train, 36/36 dev, 36/36 C y 14/14 extrapolación. En interpolación obtiene 24/40 (60 %). Coincide con 51/54 y 53/54 predicciones de los modelos en D. Los pares excluyen ignorar la pregunta, pero permiten este atajo que usa ambos números sin implementar `>=`. La coincidencia no identifica el algoritmo interno.
- **Sondas exploratorias nuevas, CPU:** 85 filas por checkpoint con cinco umbrales, 17 valores y distractores constantes. Reproducción previa de los 90 argmax históricos C+D sin discrepancias en ambas semillas. Se observan respuestas no monótonas al aumentar exclusivamente el valor: ambos responden Sí para 65≥60 y No para 75≥60. Aciertos agregados 55/85 y 57/85; muestra diseñada tras inspeccionar E25, no un benchmark independiente. Por tanto, el fallo no se limita a umbrales desconocidos ni queda atribuido únicamente a densidad de rejilla.
- Siguiente intervención recomendada, no ejecutada: romper el atajo de pertenencia a decena con datos que cubran orden dentro y entre decenas, positivos cercanos y lejanos, negativos en la misma decena y pares/control de monotonía. Comparar con control de presupuesto y cantidad de filas; crear un test nuevo separado de los conjuntos ya inspeccionados. La densidad de umbrales es una hipótesis, no una causa demostrada. Olvido y suite siguen pendientes.
- Artefactos: `runs/kaggle-typed-e25/` (hardware.json, preflight, **dos checkpoints por semilla: inicial/final**; parada anterior a e025; curvas, 84 predicciones train/dev por semilla, tests C/D), `runs/e25-analysis/`, `runs/e25-prep/`. 18 hashes verificados. Revisión y sondas en `runs/e25-review-2026-10-10/`; informe corregido v2 separado, históricos intactos. Tests pertinentes de la revisión 32/32. Commits del cierre: `66e51957`, `ab8061ee`, `99e8d4b3`.

## E24 — Prueba mínima de AJUSTE numérico (kernel mokka-typed-e24 v1) — COMPLETADO

- Objetivo diagnóstico (no generalización): ¿puede JuliaArch-small AJUSTAR 48 problemas con un único campo numérico (`motor_temp_c`), un operador (`>=`, «at least»), un umbral (60) y pregunta directa No/Sí? El gradiente dominante de las tareas originales queda ELIMINADO (solo se entrena con el microconjunto). Protocolo congelado antes de entrenar; encargo tras la revisión E23 (los checkpoints E23 fallaban incluso en las 656 filas numéricas VISTAS).
- Datos: 48 train (16 pares de inversión que difieren SOLO en el valor + frontera exacta 60.0→Sí + near-boundaries) · 32 dev balanceado con valores disjuntos · tres contrasts condicionales: 64 valores nuevos, 48 filas con umbrales nuevos (45/75/90, fronteras incluidas), 64 filas con el campo renombrado (`bearing_temp_c`, mismos valores). Este último también cambia `equipment_id` y `site` en las 64 filas: no es un contraste puro del nombre. **256/256 etiquetas re-derivadas por verificador independiente** (regex sobre pregunta + parseo de estado); serialización sin truncamientos verificada a nivel de tokens; 21 tests con literales calculados a mano (suite 119/119 en el cierre; revisión posterior: 21/21 tests E24).
- Preflight en minibatch real, todo verde: loss finita 0,690≈ln2, autograd activo, gradientes no nulos en encoder (0,52) Y cabeza (0,37), pesos cambian tras optimizer.step, correspondencia ejemplos↔logits↔targets (lote vs fila aislada, 3,3e-07), recarga del inicial verificada por hash.
- **Veredicto: FIT_CONFIRMED (2/2 semillas).** Primer train perfecto en los índices de época 24/20 (conteo desde cero: épocas 25/21); parada tras 30/23 épocas. El contador registra 60/46 intentos de paso frente a 300 presupuestados; no registra cuántas actualizaciones omitió AMP. Train 1,000 y dev 1,000 en ambas. s42 tiene norma de gradiente del encoder `inf` en tres entradas (índices 21, 22 y 26), s43 ninguna. No se guardaron escala AMP ni pasos omitidos, por lo que no se acredita que fueran saltos normales. El scheduler avanza incondicionalmente tras cada intento; registrar y distinguir estas cantidades en el siguiente runner.
- **Contrasts separados (checkpoint final, sin cherry-picking):**

| contraste | s42 | s43 | lectura |
|---|---|---|---|
| valores nuevos (64) | **1,000** | **1,000** | 64/64 aciertos en esta muestra con umbral fijo; 16/16 pares ambos-correctos |
| campo renombrado (64) | **1,000** | **1,000** | 64/64 aciertos con un único campo numérico; no prueba selección del campo entre varios |
| umbrales nuevos (48) | 0,500 | 0,500 | 0/12 pares; **48/48 predicciones coinciden con `value ≥ 60`** en cada semilla |

- Interpretación: el sistema puede ajustar esta clasificación con umbral fijo y acertar los valores reservados. Entrenar con un solo umbral no permite distinguir aprender una frontera fija de aprender a comparar valor y umbral variables. Las respuestas a umbrales nuevos son compatibles con la frontera 60 (16 No/32 Sí), pero no revelan por sí solas el mecanismo interno. Tampoco queda resuelta la vinculación campo-valor: hay un solo campo numérico y no hace falta elegirlo entre alternativas. El ajuste descarta una incapacidad absoluta de ajustar este microconjunto; no exonera arquitectura/formato para tareas más exigentes ni establece la causa de E23.
- Experimento de olvido PREPARADO, NO lanzado (E25: reanudar el checkpoint ajustado con tareas originales mezcladas y medir el microconjunto antes/después).
- Artefactos: `runs/kaggle-typed-e24/` (3 checkpoints por semilla: inicial/e015/final; curvas por época; predicciones 80 filas/semilla; evals por fila y preflight), `runs/e24-analysis/`, `runs/e24-prep/`; 21 hashes verificados en la revisión. Log descargado vacío; el runner contiene guard CUDA, pero no se conservó telemetría de hardware en el resumen. Consumo comunicado al cierre: 187,8 s de cuota, no reconciliado aquí con la fuente remota. Commits: `caadcc0` (revisión E23), `f65bfb9f` (prep), `ca4644f4` (cierre).
- Siguiente contraste recomendado: aprender con umbrales variables y pares que mantengan exactamente el mismo estado mientras cambian solo el umbral de la pregunta y fuerzan invertir la respuesta. Separar valores nuevos a umbrales vistos de umbrales reservados. Estudiar después la retención; ningún entrenamiento adicional ejecutado en esta revisión.

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

## E22 — Batches equivalentes + contraste de formato — ejecución completada, runtime pendiente de corregir

- Integrada la revisión E21: retractadas «primera medición homogénea» y «1,5×» (commit 1ff5643); conservadas las mediciones antiguas con sus asimetrías.
- **Eficiencia v2 histórica** (`runs/e22-efficiency/`): batch 1 y 16 para los tres modelos, con estado propio por fila; mismas 750 preguntas, 2 invocaciones por modelo en orden registrado (julia→ctrl→cand→cand→ctrl→julia), duraciones crudas y tamaño real del último lote (14), **cero discrepancias de predicción en las 6 invocaciones**. El harness usa el collator y llamadas directas a `engine.model` para separar fases; no usa el contexto de inferencia de `engine.logits`. La revisión detectó que los 8 hilos registrados eran una constante y no la configuración efectiva después de cargar el modelo. Conservar las cifras siguientes como históricas, sin atribuir todavía una ventaja de inferencia:

| modelo | carga | b1 p50/p95 ms (latencia individual) | b16 p50 ms (duración LOTE de 16) | thr b1 / b16 |
|---|---|---|---|---|
| Julia oficial | 5,6 s | 86,6 / 119,3 | 1.750,8 | 11,1 / 9,6 preg/s |
| Mokka ctrl_s43 | 6,1 s | 64,0 / 87,9 | 1.180,7 | 14,9 / 14,2 preg/s |
| Mokka cand_s43 | 5,8 s | 64,4 / 88,2 | 1.184,3 | 14,8 / 14,1 preg/s |

- **Corrección del runtime (2026-10-10):** el cargador Julia llama a `configure`, que vuelve a fijar hilos desde `JULIA_CPU_THREADS` (4 por defecto), después del `set_num_threads(8)` del harness. En la reproducción local sin esa variable, el número efectivo cambia 8→4. Las ejecuciones históricas no registraron el número efectivo ni el entorno, por lo que no se puede recuperar con certeza su configuración. Además, las llamadas directas a ambos modelos se hicieron sin `no_grad`/`inference_mode`; `eval()` no desactiva autograd. Se retiran las conclusiones «1,35×» y «1,48×» como ventaja de inferencia homogénea hasta repetir la medición. Memoria de proceso sigue **no medida**; tamaño de archivo de pesos no la sustituye.
- El harness revisión 3 fija los hilos después de cargar, registra el número efectivo y modo de ejecución, iguala precisión de matmul y pesos float32, usa `torch.inference_mode()` y saca el control de finitud del intervalo cronometrado para ambos. Prueba de regresión con un módulo pequeño: PASS. **La revisión 3 todavía no se ha medido con los checkpoints completos.** Los sources v2 se archivaron antes de editar; las duraciones históricas permanecen intactas.
- **Batería de formato** (`data/e22-format/`): 352 filas y contenidos distintos, 176 true/176 false, balance por operador/formato y etiquetas 352/352 re-verificadas. El baseline constante correcto es **0,500 para todos**. La columna histórica 0,764/0,676/1,000 calculaba frecuencia mayoritaria de PREDICCIONES (aquí coincide con true-rate), no acierto contra el oro. Accuracy global sin cambios: Julia **0,4972**, ctrl_s43 **0,5170**, cand_s43 **0,5000**. Pares de inversión: **0/12** ambos-correctos en cada formato y modelo.
- Los formatos probados no rescatan el rendimiento en estas sondas: accuracy por formato 0,477–0,557. Eso no demuestra invariancia o equivalencia estadística entre formatos; no se habían calculado los intervalos anunciados en el docstring. Para remuestreo pareado, mantener juntos ambos miembros de cada inversión y sus formatos (`pair` si existe, si no `group_id`): 76 unidades en vez de 88 miembros separados. Extracción numérica, vinculación campo-valor y comparación siguen abiertas; la redacción concreta de las plantillas tampoco queda descartada por evaluar solo cuatro formatos.
- **Propuesta de entrenamiento preparada y NO lanzada** (`docs/internal/E22_TRAINING_PROPOSAL.md`): control (mezcla aleatoria) vs candidato (mismo multiconjunto ordenado por fases), inicialización/pasos/optimizador idénticos por semilla, evaluación atómica nueva sin fugas (umbrales/valores nuevos), criterios de éxito que superan el «>50 % / pares > 0» (Δ≥+15 pp vs control, pares ≥0,50, límites de regresión 2 pp) y declaración de qué NO probaría un fracaso.
- **Revisión local:** corregidos baseline y etiquetas del informe, añadido análisis desde predicciones guardadas sin sobrescribirlas, y agrupación de pares declarada correctamente. Informe corregido: `runs/e22-review-2026-10-10/format_report_v2.json`; evidencia: `verification.json`. Pasan 11 pruebas pertinentes (6 existentes y 5 nuevas). El total histórico del agente era 87/87; no se repitió toda la suite. Sin entrenamiento ni GPU.

## E23 — Aprendizaje numérico + orden por bloques (kernel mokka-typed-e23 v3) — ejecución completada, interpretación corregida

- Dos preguntas previstas: **A**, adquisición con transferencia; **B**, efecto del orden. Ejecución real: 6.656 filas por brazo = 4.950 originales + **1.376 nuevas en total** + 330 de retención. Las nuevas son **640 numéricas base + 16 miembros/formato de 4 pares + 240 categóricas + 480 AND/OR**. Por tanto, hubo 656 filas numéricas, no 1.376 numéricas más otros bloques. Categóricas: 9 contenidos repetidos en 240 filas; AND/OR: 24 contenidos repetidos en 480. Inicializaciones emparejadas 3/3, 624 pasos, tres semillas por receta. Test: 528 preguntas con torque_nm, salinity_ppt y altitude_m, cuatro operadores y 12 pares con dos ubicaciones de regla. Etiquetas 2.064/2.064 re-verificadas.
- **Desviaciones del protocolo escrito:** indicaba 7.650 filas, 720 pasos, dev 860 y test 336; lo ejecutado fue 6.656/624/730/528. El candidato presenta todas las 1.376 filas nuevas, después las originales y por último retención, repitiendo ese orden cada época. No intercala originales entre fases como describía el protocolo. El control repite una permutación fija por semilla. Se mantienen los artefactos y el protocolo históricos; no reescribirlos retrospectivamente como si coincidieran.
- **Defecto de selección:** los 160 ejemplos numéricos del dev son TODOS true. El dev real combina esos 160 con 570 originales y no incluye el dev-retención anunciado. Responder siempre true obtiene 100 % en el nuevo bloque; la accuracy de ese bloque es idéntica a su true-rate. La selección de checkpoints está afectada y no permite diagnosticar capacidad numérica equilibrada. No cambiar el dev después del resultado y conservar la etiqueta de selección congelada.
- Ejecución: 2 incidentes menores corregidos (import order, doble conversión) → v3 COMPLETE con GPU 2×T4 verificada; ~52 min de pared.

| test (s42/s43/s44) | accuracy | pares ambos-correctos, 24 instancias par-formato |
|---|---|---|
| e23_mix | 0,504/0,500/0,509 | 0,167/0,000/0,083 |
| e23_cur | 0,500/0,500/0,496 | 0,000/0,000/0,083 |
| e20_ctrl_s43 (referencia) | 0,502 | 0,042 |
| Julia | 0,496 | 0,000 |

- **A: adquisición generalizable no demostrada por los checkpoints seleccionados.** El mejor promedio de receta supera al checkpoint previo +0,25 pp, lejos del criterio +15 pp; tampoco cumple los pares exigidos. Accuracy cercana al baseline 0,50, no «exactamente azar» en todos los casos. El defecto del dev y las desviaciones impiden presentar esto como una prueba limpia de incapacidad arquitectónica o de que los datos atómicos no puedan enseñar la habilidad.
- **B: INCONCLUSO.** Diferencias cur−mix −0,38/0,00/−1,33 pp; media **−0,57 pp**, IC95 corregido por problema **[−2,18, +1,08] pp**, condicional a los tres pares observados. Dispersión de diferencias entre semillas: sd 0,68 pp. Dev global mix 0,666–0,736 y cur 0,586–0,623, con el sesgo de selección descrito. La retención prevista en dev E20 no fue evaluada; no sustituirla por el test numérico nuevo.
- **Corrección del analizador:** el bootstrap sobrescribía semillas con `dict.update`, conservando solo la última. Ahora promedia diferencias dentro de cada grupo sobre todas las semillas y conserva denominadores de preguntas al remuestrear grupos de tamaños distintos. El cálculo de pares sobrescribía la ubicación `options` con `question`. Ahora conserva ambas: mix_s42 tiene **4/12** pares correctos en options y 0/12 en question; mix_s44 2/12 y 0/12; cur_s44 0/12 y 2/12; el checkpoint anterior ctrl_s43 1/12 y 0/12. No había un único modelo con pares acertados. Ninguno demuestra una mejora robusta por estos pocos aciertos.
- **Diagnóstico adicional sobre ejemplos VISTOS (CPU, sin entrenar, posterior al test):** evaluados los seis checkpoints seleccionados sobre las 656 filas numéricas de entrenamiento, con etiquetas equilibradas. Mix: **0,5030/0,5000/0,5107**; cur: **0,5000/0,5000/0,5030**; referencia anterior ctrl_s43 **0,5030**. Estos checkpoints fallan también en ejemplos vistos, por lo que el problema no se limita a rasgos nuevos. Esto no reconstruye capacidades de épocas descartadas ni prueba que nunca se aprendiera temporalmente. Artefactos en `runs/e23-review-2026-10-10/seen-numeric/`.
- Siguiente diagnóstico recomendado: demostrar ajuste de un conjunto numérico pequeño y equilibrado, con seguimiento separado de loss/acierto numéricos y dev equilibrado, antes de otro experimento grande. Separar después valores nuevos, umbrales nuevos y rasgos nuevos. No atribuir sin medición el fallo a dominancia del gradiente, JSON, arquitectura o número de épocas.
- Artefactos históricos: `runs/kaggle-typed-e23/` y `runs/e23-sealed-eval/`; 30 hashes verificados. Revisión: `runs/e23-review-2026-10-10/e23_report_v2.json` y `verification.json`. Originales conservados. Once pruebas pertinentes pasan (6 nuevas del analizador E23 y 5 de la revisión E22); el 92/92 del agente era el total histórico, no una nueva ejecución completa. Sin entrenamiento nuevo ni GPU. La medición de eficiencia revisión 3 sigue pendiente: no existe salida local en `runs/e22-efficiency-v3/`.
