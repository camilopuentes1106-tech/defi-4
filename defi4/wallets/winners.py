"""Reglas compartidas para clasificar y seleccionar perfiles ganadores POL.

No hace llamadas de red ni depende de Alchemy. Ambos pipelines usan este
módulo para que la definición de ``winner`` no se desvíe entre informes.
"""

from __future__ import annotations

from typing import Iterable

import pandas as pd


MIN_DECISIONES_GANADORA = 3
MIN_CONSISTENCY_SCORE = 0.60
MAX_PROFIT_FACTOR = 3.0


def calcular_consistency_score(
    tasa_acierto: float,
    profit_factor: float,
    n_decisiones: int,
) -> float:
    """Combina acierto, calidad PnL y cantidad de evidencia en un score 0..1."""
    evidence = min(max(n_decisiones, 0) / 10.0, 1.0)
    bounded_profit_factor = min(max(profit_factor, 0.0), MAX_PROFIT_FACTOR)
    return 0.50 * tasa_acierto + 0.30 * (bounded_profit_factor / MAX_PROFIT_FACTOR) + 0.20 * evidence


def clasificar_winner_status(
    *,
    n_decisiones: int,
    n_pendientes: int,
    pnl_neto_usdc: float,
    retorno_neto_mediano: float,
    consistency_score: float,
    min_decisiones: int = MIN_DECISIONES_GANADORA,
    min_consistency_score: float = MIN_CONSISTENCY_SCORE,
) -> str:
    """Devuelve ``pending``, ``insufficient``, ``winner`` o ``not_winner``."""
    if min_decisiones < 1:
        raise ValueError("min_decisiones debe ser >= 1.")
    if n_pendientes > 0 and n_decisiones == 0:
        return "pending"
    if n_decisiones < min_decisiones:
        return "insufficient"
    if (
        pnl_neto_usdc > 0
        and retorno_neto_mediano > 0
        and consistency_score >= min_consistency_score
    ):
        return "winner"
    return "not_winner"


def filtrar_wallets_ganadoras(
    perfiles: pd.DataFrame,
    *,
    acciones: Iterable[str] | None = None,
    horizontes: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Selecciona perfiles ya clasificados como ganadores y los ordena.

    Una wallet puede aparecer más de una vez porque es ganadora por la
    combinación concreta de acción y horizonte, por ejemplo ``BUY_POL / 1h``.
    """
    required = {"wallet", "winner_status"}
    missing = required.difference(perfiles.columns)
    if missing:
        raise ValueError(f"Faltan columnas de perfiles: {sorted(missing)}")
    winners = perfiles[perfiles["winner_status"] == "winner"].copy()
    if acciones is not None:
        if "accion" not in winners.columns:
            raise ValueError("No existe la columna accion.")
        winners = winners[winners["accion"].isin(tuple(acciones))]
    if horizontes is not None:
        if "horizonte" not in winners.columns:
            raise ValueError("No existe la columna horizonte.")
        winners = winners[winners["horizonte"].isin(tuple(horizontes))]
    order = [column for column in ("consistency_score", "pnl_neto_usdc", "n_decisiones") if column in winners.columns]
    return winners.sort_values(order, ascending=[False] * len(order), na_position="last").reset_index(drop=True) if order else winners.reset_index(drop=True)


def resumen_estados_perfiles(perfiles: pd.DataFrame) -> dict[str, int]:
    """Cuenta estados para explicar por qué una cohorte todavía no tiene ganadoras."""
    if perfiles.empty or "winner_status" not in perfiles.columns:
        return {}
    return {str(status): int(count) for status, count in perfiles["winner_status"].value_counts().items()}
