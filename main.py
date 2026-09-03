"""Entrada de consola de la librería DEFI IV.

Ejemplos:
    python main.py parquet --parquet cache/swaps_pol_usdc_24h.parquet
    python main.py rl --snapshot-dir informes_pol/snapshot_...
"""

from defi4.pipeline.wallets import main


if __name__ == "__main__":
    raise SystemExit(main())
