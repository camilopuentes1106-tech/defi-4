# Presentación final DEFI IV: Bellman, Q-learning y DQN

El archivo principal es `Presentacion_Sustentacion_DEFI_IV.tex`. Sube todos los archivos de esta carpeta a un proyecto de Overleaf y selecciona ese archivo como documento principal. El proyecto incluye el bloque DQN, los resultados y las figuras necesarias para compilar.

El bloque DQN utiliza los resultados del cache `cache/swaps_24h.parquet` del repositorio defi-4. Incluye un replay común de las horas completas del snapshot (dentro de muestra), y una evaluación separada de Bellman, Q-learning, DQN y Buy and Hold en cuatro horas reservadas, presentada como tabla en el anexo. Los modelos del replay completo se entrenan en ese mismo período y son distintos de los modelos del test.

Para regenerar las gráficas DQN, ejecuta desde la raíz del repositorio:
```bash
python presentacion_final_dqn/generar_figuras_dqn.py
```

Se requieren las dependencias del proyecto y el archivo de cache. El script recalcula las métricas y actualiza `resultados_dqn.tex` y las figuras. No se deben considerar los resultados del test de cuatro horas como evidencia suficiente de rentabilidad futura.


Se incluyen dos diapositivas sobre las acciones y las transiciones del DQN en train y test. Las decisiones de train son las registradas durante episodios repetidos; el test tiene solo cuatro decisiones. Para regenerar esas cifras se necesita `lib_rl/explicacion_dqn.py` del repositorio.
