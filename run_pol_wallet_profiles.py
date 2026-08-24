"""Punto de entrada del flujo local de perfiles POL.

Ejemplo:
    python run_pol_wallet_profiles.py --parquet /content/swaps_24h.parquet --output-dir /content/informes_pol

No descarga bloques ni consulta servicios externos. Requiere que estén en la
misma carpeta ``pol_wallet_report_from_parquet.py``, ``pol_wallet_winners.py``
y ``pol_wallet_summary.py``.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import pandas as pd

from pol_wallet_report_from_parquet import generar_informe_desde_parquet
from pol_wallet_winners import filtrar_wallets_ganadoras, resumen_estados_perfiles


def construir_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Genera perfiles Buy/Sell/Hold y un informe HTML desde un Parquet de swaps POL.",
    )
    parser.add_argument(
        "--parquet", required=True,
        help="Ruta al swaps_*.parquet o a la carpeta que lo contiene.",
    )
    parser.add_argument(
        "--output-dir", default="data/derived/parquet_wallet_report",
        help="Carpeta donde se escribirá el nuevo snapshot del informe.",
    )
    parser.add_argument(
        "--min-decisiones", type=int, default=3,
        help="Decisiones maduras mínimas para clasificar un perfil como ganador (predeterminado: 3).",
    )
    parser.add_argument(
        "--png", action="store_true",
        help="Exporta también PNG; requiere kaleido. El HTML siempre se crea.",
    )
    parser.add_argument(
        "--tabla-wallets", default=None,
        help="Parquet de WalletView.df para el ranking agregado Buy/Sell por horizonte.",
    )
    return parser


def ejecutar_flujo(
    *,
    parquet_path: str | Path,
    output_dir: str | Path,
    min_decisiones: int = 3,
    export_png: bool = False,
    tabla_wallets: str | Path | None = None,
) -> tuple[Path, pd.DataFrame]:
    """Ejecuta el informe y devuelve su carpeta y tabla de perfiles ganadores."""
    result = generar_informe_desde_parquet(
        parquet_path=parquet_path,
        output_dir=output_dir,
        min_decisiones=min_decisiones,
        export_png=export_png,
        tabla_wallets=tabla_wallets,
    )
    profiles = pd.read_parquet(result / "perfiles_wallet.parquet")
    return result, filtrar_wallets_ganadoras(profiles)


def main(argv: Sequence[str] | None = None) -> int:
    args = construir_parser().parse_args(argv)
    result, winners = ejecutar_flujo(
        parquet_path=args.parquet,
        output_dir=args.output_dir,
        min_decisiones=args.min_decisiones,
        export_png=args.png,
        tabla_wallets=args.tabla_wallets,
    )
    profiles = pd.read_parquet(result / "perfiles_wallet.parquet")
    states = resumen_estados_perfiles(profiles)

    print(f"\nInforme: {result / 'informe_wallets.html'}")
    print(f"Perfiles ganadores: {len(winners)}")
    print("Estados: " + (", ".join(f"{name}={count}" for name, count in states.items()) or "sin perfiles"))
    if not winners.empty:
        columns = [column for column in (
            "wallet", "accion", "horizonte", "consistency_score", "pnl_neto_usdc", "n_decisiones",
        ) if column in winners.columns]
        print("\nGanadoras:")
        print(winners[columns].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
