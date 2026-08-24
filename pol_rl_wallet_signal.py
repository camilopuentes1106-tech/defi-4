"""Señales causales de wallets para el MDP de trading POL.

El módulo utiliza exclusivamente perfiles Buy/Sell/Hold de horizonte 1h con
``winner_status='winner'`` y ``consistency_score >= 0.80``.  Las funciones por
corte recalculan los perfiles usando sólo decisiones cuyo precio forward ya era
conocido en ese corte; por ello no filtran con información futura.
"""

from __future__ import annotations

from datetime import datetime
from typing import Iterable, Mapping

import numpy as np
import pandas as pd

HORIZONTE_WALLET_RL = "1h"
CONSISTENCY_THRESHOLD_RL = 0.80
ACCIONES_RL = ("BUY_POL", "SELL_POL", "HOLD")
_DIRECCION_AGENTE = {"BUY_POL": "BUY", "SELL_POL": "SELL", "HOLD": "HOLD"}


def _as_utc(value: datetime | pd.Timestamp | str) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    return timestamp.tz_localize("UTC") if timestamp.tzinfo is None else timestamp.tz_convert("UTC")


def _validar_umbral(consistency_threshold: float) -> float:
    threshold = float(consistency_threshold)
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("consistency_threshold debe estar entre 0 y 1.")
    return threshold


def _profit_factor(values: pd.Series) -> float:
    positive = float(values.clip(lower=0).sum())
    negative = float((-values.clip(upper=0)).sum())
    if negative == 0:
        return 3.0 if positive > 0 else 0.0
    return min(positive / negative, 3.0)


def seleccionar_wallets_dirigidas_1h(
    perfiles: pd.DataFrame,
    *,
    consistency_threshold: float = CONSISTENCY_THRESHOLD_RL,
) -> pd.DataFrame:
    """Selecciona las wallets de alta consistencia para guiar al agente.

    Una fila representa una combinación ``wallet × acción × 1h``. No incluye
    candidatas ni perfiles insuficientes: si no hay evidencia confirmada, el
    estado RL debe recibir la señal ``NEUTRAL``.
    """
    threshold = _validar_umbral(consistency_threshold)
    required = {"wallet", "accion", "horizonte", "winner_status", "consistency_score"}
    missing = required.difference(perfiles.columns)
    if missing:
        raise ValueError("Faltan columnas de perfiles: " + ", ".join(sorted(missing)))
    selected = perfiles[
        perfiles["accion"].isin(ACCIONES_RL)
        & perfiles["horizonte"].eq(HORIZONTE_WALLET_RL)
        & perfiles["winner_status"].eq("winner")
        & (pd.to_numeric(perfiles["consistency_score"], errors="coerce") >= threshold)
    ].copy()
    selected["wallet"] = selected["wallet"].astype(str).str.lower()
    selected["direccion_agente"] = selected["accion"].map(_DIRECCION_AGENTE)
    order = [column for column in ("consistency_score", "pnl_neto_usdc", "n_decisiones") if column in selected]
    return selected.sort_values(order, ascending=[False] * len(order), na_position="last").reset_index(drop=True)


def calcular_wallets_ganadoras_1h_consistentes(
    perfiles: pd.DataFrame,
    *,
    consistency_threshold: float = CONSISTENCY_THRESHOLD_RL,
) -> pd.DataFrame:
    """Calcula la cohorte de wallets que puede dirigir el RL en 1 hora.

    El umbral solicitado es ``0.80`` por defecto. Una wallet entra sólo si ya
    es ganadora según las reglas del perfil (mínimo de decisiones maduras,
    PnL y retorno mediano positivos) y su ``consistency_score`` alcanza ese
    nivel. El resultado conserva la acción dirigida: ``BUY``, ``SELL`` o
    ``HOLD``.
    """
    return seleccionar_wallets_dirigidas_1h(
        perfiles, consistency_threshold=consistency_threshold,
    )


def construir_perfiles_causales_1h(
    ledger: pd.DataFrame,
    *,
    as_of: datetime | pd.Timestamp | str,
    consistency_threshold: float = CONSISTENCY_THRESHOLD_RL,
    min_decisiones: int = 3,
) -> pd.DataFrame:
    """Calcula perfiles 1h conocidos hasta ``as_of`` para evitar fuga temporal."""
    threshold = _validar_umbral(consistency_threshold)
    if min_decisiones < 1:
        raise ValueError("min_decisiones debe ser >= 1.")
    required = {
        "wallet", "accion", "horizonte", "decision_time", "forward_time",
        "retorno_neto", "ganancia_neta_usdc",
    }
    missing = required.difference(ledger.columns)
    if missing:
        raise ValueError("Faltan columnas del ledger: " + ", ".join(sorted(missing)))
    cutoff = _as_utc(as_of)
    frame = ledger.copy()
    frame["decision_time"] = pd.to_datetime(frame["decision_time"], utc=True, errors="coerce")
    frame["forward_time"] = pd.to_datetime(frame["forward_time"], utc=True, errors="coerce")
    frame = frame[
        frame["accion"].isin(ACCIONES_RL)
        & frame["horizonte"].eq(HORIZONTE_WALLET_RL)
        & frame["decision_time"].notna()
        & frame["forward_time"].notna()
        & (frame["decision_time"] <= cutoff)
    ].copy()
    columns = [
        "wallet", "accion", "horizonte", "n_decisiones", "n_pendientes",
        "pnl_neto_usdc", "retorno_neto_mediano", "tasa_acierto",
        "profit_factor", "consistency_score", "winner_status",
    ]
    if frame.empty:
        return pd.DataFrame(columns=columns)

    keys = ["wallet", "accion"]
    totals = frame.groupby(keys, as_index=False).size().rename(columns={"size": "n_total"})
    completed = frame[
        (frame["forward_time"] <= cutoff)
        & frame["retorno_neto"].notna()
        & frame["ganancia_neta_usdc"].notna()
    ].copy()
    if completed.empty:
        summary = totals.assign(
            n_decisiones=0, pnl_neto_usdc=np.nan, retorno_neto_mediano=np.nan,
            tasa_acierto=np.nan, profit_factor=np.nan, consistency_score=np.nan,
        )
    else:
        completed["acierto"] = (completed["retorno_neto"] > 0).astype(float)
        completed["pnl_positivo"] = completed["ganancia_neta_usdc"].clip(lower=0)
        completed["pnl_negativo"] = (-completed["ganancia_neta_usdc"].clip(upper=0))
        summary = completed.groupby(keys, as_index=False).agg(
            n_decisiones=("wallet", "size"),
            pnl_neto_usdc=("ganancia_neta_usdc", "sum"),
            retorno_neto_mediano=("retorno_neto", "median"),
            tasa_acierto=("acierto", "mean"),
            ganancia_positiva=("pnl_positivo", "sum"),
            perdida_negativa=("pnl_negativo", "sum"),
        )
        summary["profit_factor"] = np.where(
            summary["perdida_negativa"] > 0,
            (summary["ganancia_positiva"] / summary["perdida_negativa"]).clip(upper=3.0),
            np.where(summary["ganancia_positiva"] > 0, 3.0, 0.0),
        )
        evidence = (summary["n_decisiones"] / 10.0).clip(lower=0.0, upper=1.0)
        summary["consistency_score"] = (
            0.50 * summary["tasa_acierto"]
            + 0.30 * (summary["profit_factor"].clip(lower=0.0, upper=3.0) / 3.0)
            + 0.20 * evidence
        )
        summary = totals.merge(summary, on=keys, how="left")
        summary["n_decisiones"] = summary["n_decisiones"].fillna(0).astype(int)
    summary["n_pendientes"] = summary["n_total"] - summary["n_decisiones"]
    winner = (
        (summary["n_decisiones"] >= min_decisiones)
        & (summary["pnl_neto_usdc"] > 0)
        & (summary["retorno_neto_mediano"] > 0)
        & (summary["consistency_score"] >= threshold)
    )
    summary["winner_status"] = np.select(
        [
            (summary["n_decisiones"] == 0) & (summary["n_pendientes"] > 0),
            summary["n_decisiones"] < min_decisiones,
            winner,
        ],
        ["pending", "insufficient", "winner"],
        default="not_winner",
    )
    summary["wallet"] = summary["wallet"].astype(str).str.lower()
    summary["horizonte"] = HORIZONTE_WALLET_RL
    return summary[columns].sort_values(
        ["winner_status", "consistency_score"], ascending=[True, False], na_position="last"
    ).reset_index(drop=True)


def calcular_consenso_wallets_1h(
    perfiles: pd.DataFrame,
    *,
    consistency_threshold: float = CONSISTENCY_THRESHOLD_RL,
) -> dict[str, float | int | str]:
    """Resume las wallets dirigidas en una señal Buy, Sell, Hold o Neutral."""
    selected = seleccionar_wallets_dirigidas_1h(
        perfiles, consistency_threshold=consistency_threshold,
    )
    support = {
        action: float(selected.loc[selected["accion"] == action, "consistency_score"].sum())
        for action in ACCIONES_RL
    }
    top = max(support.values(), default=0.0)
    leaders = [action for action, value in support.items() if np.isclose(value, top) and top > 0]
    if len(leaders) != 1:
        signal = "NEUTRAL"
        confidence = 0.0
    else:
        leader = leaders[0]
        other_support = max(value for action, value in support.items() if action != leader)
        signal = _DIRECCION_AGENTE[leader]
        confidence = (top - other_support) / (sum(support.values()) + 1e-12)
    return {
        "senal_wallets": signal,
        "confianza_wallets": float(confidence),
        "n_wallets_dirigidas": int(selected["wallet"].nunique()),
        "support_buy": support["BUY_POL"],
        "support_sell": support["SELL_POL"],
        "support_hold": support["HOLD"],
    }


def construir_senales_wallets_horarias(
    ledger: pd.DataFrame,
    cortes_horarios: Iterable[datetime | pd.Timestamp | str],
    *,
    consistency_threshold: float = CONSISTENCY_THRESHOLD_RL,
    min_decisiones: int = 3,
) -> pd.DataFrame:
    """Construye la señal causal que podrá entrar al estado del agente cada hora."""
    rows = []
    for value in cortes_horarios:
        cutoff = _as_utc(value)
        profiles = construir_perfiles_causales_1h(
            ledger, as_of=cutoff, consistency_threshold=consistency_threshold,
            min_decisiones=min_decisiones,
        )
        rows.append({"as_of": cutoff, **calcular_consenso_wallets_1h(
            profiles, consistency_threshold=consistency_threshold,
        )})
    return pd.DataFrame(rows).sort_values("as_of").reset_index(drop=True) if rows else pd.DataFrame(
        columns=["as_of", "senal_wallets", "confianza_wallets", "n_wallets_dirigidas", "support_buy", "support_sell", "support_hold"]
    )
