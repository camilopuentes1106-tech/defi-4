# Perfiles on-chain de wallets POL en Polygon

`pol_alchemy_wallet_profiles.py` es el pipeline activo para analizar wallets
de un pool Uniswap V3 POL/USDC mediante Alchemy. Descarga exclusivamente una
ventana móvil de 24 horas, en lotes de hasta 10 bloques, y conserva snapshots
solapados en disco. Esto evita un escaneo histórico completo de Polygon y
permite que las etiquetas forward de 1 h, 4 h y 1 d se vayan cerrando cuando
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
API key**. Lee el Parquet, crea cierres horarios POL/USDC a partir de los swaps
del pool, reconstruye ciclos FIFO y exporta un informe HTML con top 5 y bottom
5 legibles por actividad, volumen y duración Hold.

En Google Colab, después de subir el archivo y el módulo:

```python
# Sube juntos pol_wallet_report_from_parquet.py y pol_wallet_winners.py.
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
Parquet de swaps que quieras analizar. El informe calcula retornos 1 h, 4 h y
1 d con esos cierres del pool y etiqueta una wallet como ganadora sólo con tres
decisiones maduras, PnL y retorno mediano positivos y score ≥ 0.60. En el
último tramo de la ventana no hay cierre posterior: esas filas quedan
`PENDING`, especialmente el horizonte de 1 d con un único archivo de 24 h.

Como alternativa al notebook, `run_pol_wallet_profiles.py` es el punto de
entrada de consola:

```powershell
& .\.venv\Scripts\python.exe .\run_pol_wallet_profiles.py `
  --parquet .\data\raw\alchemy\swaps_24h.parquet `
  --output-dir .\data\derived\informes_pol
```

En Colab sube `run_pol_wallet_profiles.py`, `pol_wallet_report_from_parquet.py` y
`pol_wallet_winners.py`; después ejecuta:

```python
!python /content/run_pol_wallet_profiles.py --parquet /content/swaps_24h.parquet --output-dir /content/informes_pol
```

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
de 1 d como `PENDING`; ejecuciones posteriores reutilizan los precios y swaps
guardados para completarlas. Una wallet sólo es `winner` con al menos tres
decisiones maduras, PnL neto y retorno mediano positivos y score >= 0.60.

Hold no infiere transferencias ni balances externos: sólo mide lotes de POL
observados desde una compra hasta su venta FIFO. Sell mide la conveniencia de
vender frente a conservar POL, nunca una posición corta.

Para validar el proyecto:

```powershell
& .\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

## Módulo Dune histórico

`pol_dune_pipeline.py` sigue disponible como extractor auditable independiente
basado en Dune. No participa en el pipeline Alchemy de perfiles anterior.

## Fuente y alcance

La única fuente es la API de [Dune](https://docs.dune.com/). La extracción
parte del 4 de septiembre de 2024, cuando Polygon PoS sustituyó MATIC por POL.
POL es nativo de Polygon; en los DEX se intercambia mediante el wrapper
histórico `0x0d500b1d8e8ef31e21c99d1db9a6444d3adf1270`. El pipeline filtra por
esa dirección on-chain y no por símbolo, que puede variar entre proveedores.

| Dataset | Tabla Dune | Contenido | Justificación |
|---|---|---|---|
| `wallet_candidates` | `dex.trades` | Hasta 2 wallets por mes y por quintil de volumen proxy | Construye una cohorte pequeña, reproducible y con actividad baja, media y alta. |
| `dex_trades` | `dex.trades` | Eventos DEX crudos de cada hop de la cohorte mensual que involucra POL | Es la evidencia base de wallet iniciadora, DEX, activos, cantidades y USD. |
| `gas_fees` | `gas.fees` | Una fila de gas por transacción con un hop de POL | Conserva coste nativo y USD sin duplicarlo en swaps multipool. |
| `prices_hour` | `prices.hour` | Precio, volumen y fuente de precio por hora | Alinea posteriormente ventanas de 1h, 4h y 1d. |

`dex.trades` registra swaps multipool como varios *hops*. Para seleccionar la
cohorte, el pipeline toma el máximo `amount_usd` de los hops de cada `tx_hash`:
es sólo un *proxy* de estratificación y evita sumar erróneamente una misma ruta
varias veces. En cada quintil las wallets se ordenan por dirección y se toman
las primeras dos, de modo que la muestra es determinista y repetible. No es una
selección de "mejores" wallets ni una medición de rentabilidad.

Tras esa selección, el pipeline descarga únicamente los swaps y el gas de esas
wallets dentro del mismo mes que las hizo candidatas; el precio horario se
descarga para todo POL. Esto limita costes y volumen de Dune sin perder la
trazabilidad de cada decisión. La intención económica `BUY_POL`/`SELL_POL` y
el PnL quedan deliberadamente para la siguiente fase.

## Ejecución

1. Crear una API key de Dune con permisos de lectura.
2. Instalar dependencias:

   ```powershell
   .\.venv\Scripts\python.exe -m pip install -r requirements.txt
   ```

3. Definir la clave sólo para la sesión y ejecutar con una fecha de corte fija:

   ```powershell
   $env:DUNE_API_KEY = "tu_clave"
   & .\.venv\Scripts\python.exe .\pol_dune_pipeline.py --start 2024-09-04 --end 2026-08-17 --transport node
   ```

   En este equipo, usa `--transport node`: Node utiliza los certificados del
   sistema operativo y evita el problema TLS detectado en Python. El resultado,
   SQL y manifiesto son idénticos; sólo cambia el cliente HTTP. En otro entorno
   sin ese problema se puede omitir y usar el transporte predeterminado.

   En Google Colab:

   ```python
   !pip install -r requirements.txt
   import os
   from getpass import getpass
   os.environ["DUNE_API_KEY"] = getpass("Dune API key: ")
   %run pol_dune_pipeline.py
   snapshot_dir = run_full_pipeline()
   ```

`--end` es exclusivo; si se omite, se toma el instante UTC de ejecución. Cada
ejecución crea `data/raw/dune/snapshot_YYYYMMDDTHHMMSSZ/` y se niega a
sobrescribir una anterior.

## Auditoría y decisiones de diseño

Los resultados se dividen por mes UTC, lote de wallets y páginas de 1.000 filas
de la API para acotar memoria, tamaño, créditos de Dune y reintentos. Por cada
archivo Parquet, `manifest.json` conserva el SQL exacto, su hash, el
`execution_id` de Dune, el lote de wallets, columnas, conteo de filas, claves de
negocio duplicadas y SHA-256 del archivo. Así se puede reproducir la consulta y
detectar modificaciones posteriores.

La API key se lee de `DUNE_API_KEY`, no se imprime ni se guarda. `data/` está
en `.gitignore`: se versiona el procedimiento reproducible, no una descarga
voluminosa ni credenciales.

No se usa Alchemy en esta fase. Dune ya ofrece swaps DEX decodificados, precios
y coste de gas, que son los datos necesarios para iniciar el estudio de
comportamiento. Las transferencias genéricas añadirían pagos, contratos y otros
tokens que no demuestran una compra/venta de POL, además de una segunda fuente
que habría que conciliar. Alchemy sólo tendría sentido al investigar actividad
no DEX de una cohorte pequeña de wallets ya seleccionada.

## Referencias

- [`dex.trades`: esquema y semántica](https://docs.dune.com/data-catalog/curated/dex-trades/evm/dex-trades)
- [`prices.hour`: precio horario](https://docs.dune.com/data-catalog/curated/prices/prices_hour)
- [`gas.fees`: coste por transacción](https://docs.dune.com/data-catalog/curated/gas-fees/fees)
- [API de resultados de Dune](https://docs.dune.com/api-reference/executions/endpoint/get-execution-result)
- [Migración oficial MATIC a POL](https://docs.polygon.technology/pos/concepts/tokens/matic-to-pol)
