# Perfiles on-chain de wallets POL en Polygon

`pol_alchemy_wallet_profiles.py` es el pipeline activo para analizar wallets
de un pool Uniswap V3 POL/USDC mediante Alchemy. Descarga exclusivamente una
ventana móvil de 24 horas, en lotes de hasta 10 bloques, y conserva snapshots
solapados en disco. Esto evita un escaneo histórico completo de Polygon y
permite que las etiquetas forward de 1 m, 5 m, 15 m y 1 h se vayan cerrando cuando
el precio posterior ya es conocido.

El pipeline genera, por cada ejecución, swaps lógicos, ciclos FIFO de Hold,
un ledger Buy/Sell/Hold, perfiles de wallets, estado listo para RL y un informe
HTML con cinco gráficas más sus PNG. El resultado se escribe bajo
`data/derived/alchemy_wallet_profiles/` y el manifiesto registra los hashes de
los snapshots fuente.

## Informe inmediato desde un Parquet ya descargado

Si ya tienes `swaps_20260823T010354Z.parquet` (o el antiguo
`swaps_10h.parquet`), usa `pol_wallet_report_from_parquet.py`. Este módulo es
independiente: **no importa ni llama a Alchemy, web3, Yahoo Finance ni usa una
API key**. Lee el Parquet, toma el último precio POL/USDC del pool conocido en
cada horizonte, reconstruye ciclos FIFO y exporta un informe HTML con top 5 y bottom
5 legibles por actividad, volumen y duración Hold.

En Google Colab, después de subir el archivo y el módulo:

```python
# Sube juntos pol_wallet_report_from_parquet.py, pol_wallet_winners.py
# y pol_wallet_summary.py.
%run "/content/pol_wallet_report_from_parquet.py"

resultado = generar_informe_desde_parquet(
    parquet_path="/content/data/raw/alchemy/swaps_20260823T010354Z.parquet",
    output_dir="/content/informes_pol",
    export_png=False,
)

from google.colab import files
files.download(str(resultado / "informe_wallets.html"))
```

También puedes pasar la carpeta en lugar del archivo, siempre que contenga el
Parquet de swaps que quieras analizar. El informe calcula retornos 1 m, 5 m,
15 m y 1 h con esos precios del pool y etiqueta una wallet como ganadora sólo con tres
decisiones maduras, PnL y retorno mediano positivos y score ≥ 0.60. En el
último tramo de la ventana no hay precio posterior: esas filas quedan `PENDING`.

## Tabla WalletView: candidatas por posición y horizonte

El código de `utils/wallet.py` ya crea una tabla por `wallet × direccion` con
las columnas `ganancia_neta_usdc_1m`, `..._5m`, `..._15m` y `..._1h`. Después
de `wallets.construir()`, úsala así:

```python
# Una fila por wallet × Buy/Sell × horizonte.
candidatas = wallets.perfiles_por_horizonte(min_swaps=3)
display(candidatas[candidatas["estado_resumen"] == "candidate_winner"])

# Opcional: incorporar el ranking agregado al informe HTML.
wallets.df.to_parquet("cache/wallets_24h.parquet", index=False)
```

`candidate_winner` sólo usa la tabla agregada: mínimo tres swaps y ganancia
neta/retorno agregado positivos para esa posición y horizonte. Es una
preselección transparente. La etiqueta final `winner` sigue usando el ledger
por operación para medir tasa de acierto, mediana, profit factor y consistency
score; la tabla agregada por sí sola no contiene esa distribución.

Puedes pasar la tabla al informe sin volver a llamar a Alchemy:

```python
resultado = generar_informe_desde_parquet(
    parquet_path="cache/swaps_24h.parquet",
    tabla_wallets="cache/wallets_24h.parquet",
    output_dir="informes_pol",
    export_png=False,
)
```

Como alternativa al notebook, `run_pol_wallet_profiles.py` es el punto de
entrada de consola:

```powershell
& .\.venv\Scripts\python.exe .\run_pol_wallet_profiles.py `
  --parquet .\data\raw\alchemy\swaps_24h.parquet `
  --output-dir .\data\derived\informes_pol
```

En Colab sube `run_pol_wallet_profiles.py`, `pol_wallet_report_from_parquet.py`,
`pol_wallet_winners.py` y `pol_wallet_summary.py`; después ejecuta:

```python
!python /content/run_pol_wallet_profiles.py --parquet /content/swaps_24h.parquet --output-dir /content/informes_pol
```

## Pipeline unificado

`pol_wallet_pipeline.py` coordina los módulos sin duplicar la lógica de
wallets ganadoras. Tiene tres modos:

```powershell
# Recomendado: usa un Parquet existente, sin red ni Alchemy.
python .\pol_wallet_pipeline.py parquet --parquet .\cache\swaps_24h.parquet

# Incluye además el ranking de la tabla WalletView ya creada.
python .\pol_wallet_pipeline.py parquet --parquet .\cache\swaps_24h.parquet --tabla-wallets .\cache\wallets_24h.parquet

# Reconstruye desde snapshots ya descargados; no consulta bloques nuevos.
python .\pol_wallet_pipeline.py cache --raw-cache-dir .\data\raw\alchemy

# Sólo para una nueva extracción limitada con Alchemy.
$env:ALCHEMY_RPC_URL = "https://polygon-mainnet.g.alchemy.com/v2/TU_CLAVE"
python .\pol_wallet_pipeline.py alchemy --lookback-hours 24
```

Cada modo exporta el HTML, los Parquet derivados y una tabla final con las
wallets ganadoras. El modo `parquet` es el más rápido para colaborar porque
no vuelve a descargar la cadena.

## Ejecución con Alchemy

1. Instalar dependencias en el entorno virtual:

   ```powershell
   .\.venv\Scripts\python.exe -m pip install -r requirements.txt
   ```

2. Definir el RPC sólo para la sesión. No pegues una clave en el código ni en
   el comando:

   ```powershell
   $env:ALCHEMY_RPC_URL = "https://polygon-mainnet.g.alchemy.com/v2/TU_CLAVE"
   & .\.venv\Scripts\python.exe .\pol_alchemy_wallet_profiles.py `
     --pool-address 0xA374094527e1673A86dE625aa59517c5dE346d32 `
     --lookback-hours 24
   ```

`--as-of 2026-08-22T12:00:00Z` permite repetir un corte temporal y
`--block-span` no puede superar 10. Una primera ejecución deja las etiquetas
de 1 m, 5 m, 15 m o 1 h aún no maduras como `PENDING`; ejecuciones posteriores
reutilizan los swaps guardados para completarlas. Una wallet sólo es `winner` con al menos tres
decisiones maduras, PnL neto y retorno mediano positivos y score >= 0.60.

Hold no infiere transferencias ni balances externos: sólo mide lotes de POL
observados desde una compra hasta su venta FIFO. Sell mide la conveniencia de
vender frente a conservar POL, nunca una posición corta.

Para validar el proyecto:

```powershell
& .\.venv\Scripts\python.exe -m unittest discover -s tests -v
```
