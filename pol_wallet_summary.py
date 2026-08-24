"""Preselección de wallets desde la tabla agregada ``WalletView``.

La tabla contiene totales por ``wallet`` y ``direccion``.  Es útil para
comparar Buy y Sell en 1m, 5m, 15m y 1h, pero no guarda los retornos de cada
operación.  Por eso este módulo entrega *candidatas* y no reemplaza el
``winner_status`` del ledger detallado: tasa de acierto, mediana y profit
factor sólo se pueden calcular a partir de decisiones individuales.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


DEFAULT_HORIZONS = ("1m", "5m", "15m", "1h")
_BASE_COLUMNS = (
    "wallet",
    "direccion",
    "n_swaps",
    "pol_cantidad",
    "usdc_cantidad",
    "gas_usdc",
    "usdc_neto",
)


def _normalizar_direccion(values: pd.Series) -> pd.Series:
    """Convierte las etiquetas de ``WalletView`` al esquema del pipeline."""
    normalized = values.astype(str).str.strip().str.upper()
    return pd.Series(
        np.select(
            [normalized.isin(["BUY", "BUY_POL"]), normalized.isin(["SELL", "SELL_POL"])],
            ["BUY_POL", "SELL_POL"],
            default="AMBIGUOUS",
        ),
        index=values.index,
        dtype="object",
    )


def construir_perfiles_desde_tabla_wallets(
    tabla_wallets: pd.DataFrame,
    *,
    horizontes: Iterable[str] = DEFAULT_HORIZONS,
    min_swaps: int = 3,
) -> pd.DataFrame:
    """Convierte ``WalletView.df`` en un ranking por posición y horizonte.

    ``candidate_winner`` significa que la fila agregada tiene al menos
    ``min_swaps`` y tanto PnL como retorno agregado positivos.  No se llama
    ``winner`` porque la tabla agregada no permite medir consistencia real por
    decisión.  La clasificación final se calcula después con el ledger.
    """
    if min_swaps < 1:
        raise ValueError("min_swaps debe ser >= 1.")
    requested = tuple(horizontes)
    invalid = set(requested).difference(DEFAULT_HORIZONS)
    if invalid:
        raise ValueError(f"Horizontes no soportados: {sorted(invalid)}")
    missing = set(_BASE_COLUMNS).difference(tabla_wallets.columns)
    if missing:
        raise ValueError(
            "La tabla de wallets no tiene las columnas base: " + ", ".join(sorted(missing))
        )

    frame = tabla_wallets.copy()
    frame["wallet"] = frame["wallet"].astype(str).str.lower()
    frame["accion"] = _normalizar_direccion(frame["direccion"])
    frame = frame[frame["accion"].isin(["BUY_POL", "SELL_POL"])].copy()
    for column in _BASE_COLUMNS[2:]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    records: list[pd.DataFrame] = []
    for horizon in requested:
        pnl_column = f"ganancia_neta_usdc_{horizon}"
        part = frame[list(_BASE_COLUMNS) + ["accion"]].copy()
        part["horizonte"] = horizon
        part["pnl_neto_usdc"] = (
            pd.to_numeric(frame[pnl_column], errors="coerce")
            if pnl_column in frame.columns else np.nan
        )
        part["retorno_neto_agregado"] = np.where(
            part["usdc_neto"] > 0,
            part["pnl_neto_usdc"] / part["usdc_neto"],
            np.nan,
        )
        part["fuente_pnl"] = pnl_column if pnl_column in frame.columns else "MISSING_COLUMN"
        part["estado_resumen"] = np.select(
            [
                part["pnl_neto_usdc"].isna(),
                part["n_swaps"] < min_swaps,
                (part["pnl_neto_usdc"] > 0) & (part["retorno_neto_agregado"] > 0),
            ],
            ["missing_return", "insufficient", "candidate_winner"],
            default="not_candidate",
        )
        part["es_candidata"] = part["estado_resumen"].eq("candidate_winner")
        records.append(part)

    columns = [
        "wallet", "direccion", "accion", "horizonte", "n_swaps", "pol_cantidad",
        "usdc_cantidad", "gas_usdc", "usdc_neto", "pnl_neto_usdc",
        "retorno_neto_agregado", "estado_resumen", "es_candidata", "fuente_pnl",
    ]
    if not records:
        return pd.DataFrame(columns=columns)
    return pd.concat(records, ignore_index=True)[columns].sort_values(
        ["horizonte", "accion", "pnl_neto_usdc", "n_swaps"],
        ascending=[True, True, False, False],
        na_position="last",
    ).reset_index(drop=True)


def filtrar_candidatas_wallets(
    perfiles_resumen: pd.DataFrame,
    *,
    acciones: Iterable[str] | None = None,
    horizontes: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Filtra las candidatas agregadas, opcionalmente por acción y horizonte."""
    required = {"estado_resumen", "accion", "horizonte"}
    missing = required.difference(perfiles_resumen.columns)
    if missing:
        raise ValueError("Faltan columnas: " + ", ".join(sorted(missing)))
    result = perfiles_resumen[perfiles_resumen["estado_resumen"] == "candidate_winner"].copy()
    if acciones is not None:
        result = result[result["accion"].isin(tuple(acciones))]
    if horizontes is not None:
        result = result[result["horizonte"].isin(tuple(horizontes))]
    return result.sort_values(
        ["pnl_neto_usdc", "retorno_neto_agregado", "n_swaps"],
        ascending=[False, False, False],
        na_position="last",
    ).reset_index(drop=True)
