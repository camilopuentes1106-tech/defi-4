"""Orquestador único para los perfiles POL Buy/Sell/Hold.

Modos de uso:

    # Rápido y sin red: usa swaps ya descargados.
    python pol_wallet_pipeline.py parquet --parquet cache/swaps_24h.parquet

    # Descarga una nueva ventana limitada con Alchemy.
    python pol_wallet_pipeline.py alchemy --lookback-hours 24

    # Reconstruye un reporte desde snapshots crudos de Alchemy ya guardados.
    python pol_wallet_pipeline.py cache --raw-cache-dir data/raw/alchemy

    # Construye estados, transiciones y política Bellman sin red.
    python pol_wallet_pipeline.py rl --snapshot-dir informes_pol/snapshot_...

La lógica de perfil, selección de ganadoras y gráficas permanece en los
módulos especializados. Este archivo sólo coordina el flujo y muestra el
resultado final de forma uniforme.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import pandas as pd

from ..wallets.profiles import (
    DEFAULT_LOOKBACK_HOURS,
    DEFAULT_POOL_ADDRESS,
    ejecutar_pipeline_perfiles,
    reconstruir_perfiles_desde_cache,
)
from ..wallets.report import generar_informe_desde_parquet
from ..wallets.winners import filtrar_wallets_ganadoras, resumen_estados_perfiles
from ..model.workflow import RLPipelineResult, ejecutar_rl_desde_snapshot
from ..wallets.signals import CONSISTENCY_THRESHOLD_RL


@dataclass(frozen=True)
class PipelineResult:
    """Resultado normalizado para los tres modos del pipeline."""

    snapshot_dir: Path
    perfiles: pd.DataFrame
    ganadoras: pd.DataFrame
    estados: dict[str, int]


def _cargar_resultado(snapshot_dir: str | Path) -> PipelineResult:
    root = Path(snapshot_dir)
    profile_path = root / "perfiles_wallet.parquet"
    if not profile_path.is_file():
        raise FileNotFoundError(f"No encontré {profile_path}.")
    profiles = pd.read_parquet(profile_path)
    return PipelineResult(
        snapshot_dir=root,
        perfiles=profiles,
        ganadoras=filtrar_wallets_ganadoras(profiles),
        estados=resumen_estados_perfiles(profiles),
    )


def ejecutar_desde_parquet(
    *,
    parquet_path: str | Path,
    output_dir: str | Path = "data/derived/parquet_wallet_report",
    min_decisiones: int = 3,
    export_png: bool = False,
    tabla_wallets: pd.DataFrame | str | Path | None = None,
) -> PipelineResult:
    """Analiza un Parquet sin red y admite la tabla agregada WalletView."""
    root = generar_informe_desde_parquet(
        parquet_path=parquet_path,
        output_dir=output_dir,
        min_decisiones=min_decisiones,
        export_png=export_png,
        tabla_wallets=tabla_wallets,
    )
    return _cargar_resultado(root)


def ejecutar_desde_cache_alchemy(
    *,
    raw_cache_dir: str | Path = "data/raw/alchemy",
    output_dir: str | Path = "data/derived/alchemy_wallet_profiles",
    as_of: str | None = None,
    lookback_hours: float = DEFAULT_LOOKBACK_HOURS,
    actualizar_precios: bool = False,
    export_png: bool = False,
) -> PipelineResult:
    """Reconstruye perfiles desde caches ya guardadas; Alchemy no se consulta."""
    root = reconstruir_perfiles_desde_cache(
        raw_cache_dir=raw_cache_dir,
        output_dir=output_dir,
        as_of=as_of,
        lookback_hours=lookback_hours,
        actualizar_precios=actualizar_precios,
        export_png=export_png,
    )
    return _cargar_resultado(root)


def ejecutar_desde_alchemy(
    *,
    rpc_url: str,
    pool_address: str = DEFAULT_POOL_ADDRESS,
    lookback_hours: float = DEFAULT_LOOKBACK_HOURS,
    as_of: str | None = None,
    raw_cache_dir: str | Path = "data/raw/alchemy",
    output_dir: str | Path = "data/derived/alchemy_wallet_profiles",
    block_span: int = 10,
    pause_seconds: float = 0.05,
) -> PipelineResult:
    """Descarga una ventana limitada con Alchemy y genera perfiles y gráficas."""
    root = ejecutar_pipeline_perfiles(
        rpc_url=rpc_url,
        pool_address=pool_address,
        lookback_hours=lookback_hours,
        as_of=as_of,
        raw_cache_dir=raw_cache_dir,
        output_dir=output_dir,
        block_span=block_span,
        pausa_seg=pause_seconds,
    )
    return _cargar_resultado(root)


def ejecutar_agente_bellman_desde_snapshot(
    *,
    snapshot_dir: str | Path,
    output_dir: str | Path = "data/derived/pol_rl_bellman",
    consistency_threshold: float = CONSISTENCY_THRESHOLD_RL,
    horizon: int = 24,
    gamma: float = 0.99,
) -> RLPipelineResult:
    """Construye y resuelve el MDP desde perfiles ya generados, sin red."""
    return ejecutar_rl_desde_snapshot(
        snapshot_dir=snapshot_dir,
        output_dir=output_dir,
        consistency_threshold=consistency_threshold,
        horizon=horizon,
        gamma=gamma,
    )


def construir_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Pipeline unificado de perfiles de wallets POL.")
    commands = parser.add_subparsers(dest="modo", required=True)

    parquet = commands.add_parser("parquet", help="Genera perfiles desde un Parquet existente; no usa red.")
    parquet.add_argument("--parquet", required=True, help="Archivo swaps_*.parquet o carpeta que lo contiene.")
    parquet.add_argument("--output-dir", default="data/derived/parquet_wallet_report")
    parquet.add_argument("--min-decisiones", type=int, default=3)
    parquet.add_argument(
        "--tabla-wallets", default=None,
        help="Parquet producido desde WalletView.df; añade ranking por posición y horizonte.",
    )
    parquet.add_argument("--png", action="store_true", help="Exporta PNG además del HTML.")

    cache = commands.add_parser("cache", help="Reconstruye desde cachés Alchemy existentes; no descarga swaps.")
    cache.add_argument("--raw-cache-dir", default="data/raw/alchemy")
    cache.add_argument("--output-dir", default="data/derived/alchemy_wallet_profiles")
    cache.add_argument("--as-of", default=None)
    cache.add_argument("--lookback-hours", type=float, default=DEFAULT_LOOKBACK_HOURS)
    cache.add_argument("--actualizar-precios", action="store_true", help="Consulta Yahoo para completar precios.")
    cache.add_argument("--png", action="store_true")

    alchemy = commands.add_parser("alchemy", help="Descarga una ventana nueva con Alchemy y genera perfiles.")
    alchemy.add_argument("--rpc-url", default=os.environ.get("ALCHEMY_RPC_URL"))
    alchemy.add_argument("--pool-address", default=DEFAULT_POOL_ADDRESS)
    alchemy.add_argument("--lookback-hours", type=float, default=DEFAULT_LOOKBACK_HOURS)
    alchemy.add_argument("--as-of", default=None)
    alchemy.add_argument("--raw-cache-dir", default="data/raw/alchemy")
    alchemy.add_argument("--output-dir", default="data/derived/alchemy_wallet_profiles")
    alchemy.add_argument("--block-span", type=int, default=10)
    alchemy.add_argument("--pause-seconds", type=float, default=0.05)

    rl = commands.add_parser("rl", help="Construye el MDP/Bellman desde un snapshot existente; no usa red.")
    rl.add_argument("--snapshot-dir", required=True, help="Snapshot con perfiles_wallet, swaps_logicos y ledger_decisiones.")
    rl.add_argument("--output-dir", default="data/derived/pol_rl_bellman")
    rl.add_argument("--consistency-threshold", type=float, default=CONSISTENCY_THRESHOLD_RL)
    rl.add_argument("--horizon", type=int, default=24, help="Número de decisiones horarias de Bellman.")
    rl.add_argument("--gamma", type=float, default=0.99)
    return parser


def imprimir_resultado(result: PipelineResult) -> None:
    """Muestra la ubicación del reporte y una tabla breve de perfiles ganadores."""
    print(f"\nInforme HTML: {result.snapshot_dir / 'informe_wallets.html'}")
    print(f"Perfiles ganadores: {len(result.ganadoras)}")
    print("Estados: " + (", ".join(f"{name}={count}" for name, count in result.estados.items()) or "sin perfiles"))
    if result.ganadoras.empty:
        return
    columns = [column for column in (
        "wallet", "accion", "horizonte", "consistency_score", "pnl_neto_usdc", "n_decisiones",
    ) if column in result.ganadoras.columns]
    print("\nWallets ganadoras:")
    print(result.ganadoras[columns].to_string(index=False))


def imprimir_resultado_rl(result: RLPipelineResult) -> None:
    """Resumen breve del MDP generado para ejecución por consola."""
    print(f"\nArtefactos RL: {result.output_dir}")
    print(f"Wallets dirigidas 1h: {result.wallets_dirigidas['wallet'].nunique()}")
    print(f"Observaciones horarias: {len(result.observaciones)}")
    print("Señales: " + ", ".join(
        f"{signal}={count}" for signal, count in result.observaciones["senal_wallets"].value_counts().items()
    ))
    if not result.replay.empty:
        print(f"Riqueza final del replay histórico (base 100): {result.replay['wealth'].iloc[-1]:.4f}")


def main(argv: Sequence[str] | None = None) -> int:
    args = construir_parser().parse_args(argv)
    if args.modo == "rl":
        rl_result = ejecutar_agente_bellman_desde_snapshot(
            snapshot_dir=args.snapshot_dir,
            output_dir=args.output_dir,
            consistency_threshold=args.consistency_threshold,
            horizon=args.horizon,
            gamma=args.gamma,
        )
        imprimir_resultado_rl(rl_result)
        return 0
    if args.modo == "parquet":
        result = ejecutar_desde_parquet(
            parquet_path=args.parquet,
            output_dir=args.output_dir,
            min_decisiones=args.min_decisiones,
            export_png=args.png,
            tabla_wallets=args.tabla_wallets,
        )
    elif args.modo == "cache":
        result = ejecutar_desde_cache_alchemy(
            raw_cache_dir=args.raw_cache_dir,
            output_dir=args.output_dir,
            as_of=args.as_of,
            lookback_hours=args.lookback_hours,
            actualizar_precios=args.actualizar_precios,
            export_png=args.png,
        )
    else:
        if not args.rpc_url:
            raise SystemExit("Define ALCHEMY_RPC_URL o proporciona --rpc-url para el modo alchemy.")
        result = ejecutar_desde_alchemy(
            rpc_url=args.rpc_url,
            pool_address=args.pool_address,
            lookback_hours=args.lookback_hours,
            as_of=args.as_of,
            raw_cache_dir=args.raw_cache_dir,
            output_dir=args.output_dir,
            block_span=args.block_span,
            pause_seconds=args.pause_seconds,
        )
    imprimir_resultado(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
