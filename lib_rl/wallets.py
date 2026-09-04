"""Módulo autónomo de análisis de wallets POL: vista agregada, ciclos FIFO, ledger y perfiles.

Reconstruye el comportamiento de las wallets a partir de decisiones maduras sin fuga temporal,
clasifica wallets ganadoras por consistencia y genera señales dirigidas para el modelo Bellman.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

DEFAULT_POOL_ADDRESS = "0xA374094527e1673A86dE625aa59517c5dE346d32"
DEFAULT_LOOKBACK_HOURS = 24.0
DEFAULT_HORIZONS = ("1m", "5m", "15m", "1h")
HORIZONS: Mapping[str, pd.Timedelta] = {
    "1m": pd.Timedelta(minutes=1),
    "5m": pd.Timedelta(minutes=5),
    "15m": pd.Timedelta(minutes=15),
    "1h": pd.Timedelta(hours=1),
}

MIN_DECISIONES_GANADORA = 3
MIN_CONSISTENCY_SCORE = 0.60
MAX_PROFIT_FACTOR = 3.0
HORIZONTE_WALLET_RL = "1h"
CONSISTENCY_THRESHOLD_RL = 0.80
ACCIONES_RL = ("BUY_POL", "SELL_POL", "HOLD")
_DIRECCION_AGENTE = {"BUY_POL": "BUY", "SELL_POL": "SELL", "HOLD": "HOLD"}


@dataclass(frozen=True)
class WalletAnalysisResult:
    """Resultado del análisis completo de wallets en memoria."""

    perfiles: pd.DataFrame
    ganadoras: pd.DataFrame
    ledger: pd.DataFrame
    swaps_logicos: pd.DataFrame
    ciclos_hold: pd.DataFrame
    estados: dict[str, int]


def _as_utc(value: datetime | pd.Timestamp | str | None) -> pd.Timestamp:
    stamp = pd.Timestamp.now(tz="UTC") if value is None else pd.Timestamp(value)
    return stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")


# ── 1. Vista Agregada de Wallets (WalletView) ──────────────────────────────────

class WalletView:
    """Vista agregada por wallet y dirección de swap."""

    def __init__(self, pipeline: Any, horizontes: list[str] | None = None) -> None:
        df = getattr(pipeline, "df", pipeline) if not isinstance(pipeline, pd.DataFrame) else pipeline
        if df.empty:
            raise ValueError("El DataFrame de swaps está vacío.")
        self._pipeline = pipeline
        self._df_swaps = df
        self._horizontes = horizontes or [
            col.replace("retorno_", "") for col in df.columns if col.startswith("retorno_")
        ]
        self.df: pd.DataFrame = pd.DataFrame()

    def construir(self) -> pd.DataFrame:
        """Construye la vista agregada y la guarda en self.df."""
        df = self._df_swaps.copy()
        nombre_t0 = getattr(self._pipeline, "nombre_t0", "pol")
        nombre_t1 = getattr(self._pipeline, "nombre_t1", "usdc")

        col_t0 = f"{nombre_t0}_cantidad" if f"{nombre_t0}_cantidad" in df.columns else "pol_cantidad"
        col_t1 = f"{nombre_t1}_cantidad" if f"{nombre_t1}_cantidad" in df.columns else "usdc_cantidad"
        col_gas_t0 = f"gas_{nombre_t0}" if f"gas_{nombre_t0}" in df.columns else "gas_pol"
        col_gas_t1 = f"gas_{nombre_t1}"
        col_neto_t1 = f"{nombre_t1}_neto"

        if col_gas_t1 not in df.columns and col_gas_t0 in df.columns:
            precio = df["precio_ejecutado"] if "precio_ejecutado" in df.columns else 1.0
            df[col_gas_t1] = df[col_gas_t0] * precio

        if col_neto_t1 not in df.columns:
            if "cantidad_neta_usdc" in df.columns:
                df[col_neto_t1] = df["cantidad_neta_usdc"]
            elif col_gas_t1 in df.columns:
                df[col_neto_t1] = (df[col_t1] - df[col_gas_t1]).clip(lower=0)
            else:
                df[col_neto_t1] = df[col_t1]

        agg: dict[str, tuple[str, str]] = {
            "n_swaps": ("hash_tx", "count") if "hash_tx" in df.columns else (df.columns[0], "count"),
        }
        if col_t0 in df.columns:
            agg[col_t0] = (col_t0, "sum")
        if col_t1 in df.columns:
            agg[col_t1] = (col_t1, "sum")
        if col_gas_t1 in df.columns:
            agg[col_gas_t1] = (col_gas_t1, "sum")
        if col_neto_t1 in df.columns:
            agg[col_neto_t1] = (col_neto_t1, "sum")

        for iv in self._horizontes:
            col_ganancia = f"ganancia_{nombre_t1}_{iv}"
            if col_ganancia in df.columns:
                agg[col_ganancia] = (col_ganancia, "sum")

        group_cols = [c for c in ["wallet", "direccion"] if c in df.columns]
        resultado = df.groupby(group_cols).agg(**agg).reset_index()
        self.df = resultado
        return self.df


# ── 2. Swaps Lógicos ──────────────────────────────────────────────────────────

def canonicalizar_swaps_logicos(swaps: pd.DataFrame) -> pd.DataFrame:
    """Agrupa eventos multihop por transacción y calcula cantidades netas y gas en USDC."""
    if swaps.empty:
        return pd.DataFrame()
    frame = swaps.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    if "wallet" in frame.columns:
        frame["wallet"] = frame["wallet"].astype(str).str.lower()

    col_pol = "pol_cantidad" if "pol_cantidad" in frame.columns else "cantidad_pol"
    col_usdc = "usdc_cantidad" if "usdc_cantidad" in frame.columns else "cantidad_usdc"
    frame["_signed_pol"] = np.where(frame["direccion"].astype(str).str.lower().isin(["buy", "buy_pol"]), frame[col_pol], -frame[col_pol])
    frame["_signed_usdc"] = np.where(frame["direccion"].astype(str).str.lower().isin(["sell", "sell_pol"]), frame[col_usdc], -frame[col_usdc])

    group_keys = ["hash_tx", "wallet"]
    records = []
    for (tx_hash, wallet), group in frame.groupby(group_keys):
        first = group.iloc[0]
        net_pol = float(group["_signed_pol"].sum())
        net_usdc = float(group["_signed_usdc"].sum())
        gas_pol = float(first.get("gas_pol", 0.0))
        gas_usdc = float(first.get("gas_usdc", 0.0))

        if abs(net_pol) < 1e-9:
            continue
        accion = "BUY_POL" if net_pol > 0 else "SELL_POL"
        pol_abs = abs(net_pol)
        usdc_abs = abs(net_usdc)
        precio_ejecutado = usdc_abs / pol_abs if pol_abs > 0 else float(first.get("precio_ejecutado", np.nan))

        if gas_usdc <= 0 and gas_pol > 0 and not np.isnan(precio_ejecutado):
            gas_usdc = gas_pol * precio_ejecutado

        records.append({
            "timestamp": first["timestamp"],
            "bloque": first.get("bloque", 0),
            "hash_tx": tx_hash,
            "wallet": wallet,
            "accion": accion,
            "pol_cantidad": pol_abs,
            "usdc_cantidad": usdc_abs,
            "precio_ejecutado": precio_ejecutado,
            "gas_pol": gas_pol,
            "gas_usdc": gas_usdc,
        })

    canonical = pd.DataFrame(records)
    if canonical.empty:
        return canonical
    return canonical.sort_values("timestamp").reset_index(drop=True)


# ── 3. Precios de Referencia y Ciclos Hold FIFO ───────────────────────────────

def construir_precios_por_minuto_desde_swaps(swaps: pd.DataFrame) -> pd.Series:
    """Construye una serie temporal por minuto de precios de cierre para evaluar decisiones."""
    frame = swaps.dropna(subset=["timestamp", "precio_ejecutado"]).copy()
    if frame.empty:
        return pd.Series(dtype=float)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True).dt.floor("1min")
    series = frame.groupby("timestamp")["precio_ejecutado"].last().sort_index()
    idx = pd.date_range(series.index.min(), series.index.max(), freq="1min", tz="UTC")
    return series.reindex(idx).ffill().bfill()


def construir_ciclos_hold(swaps: pd.DataFrame) -> pd.DataFrame:
    """Empareja compras y ventas por wallet mediante FIFO para delimitar períodos de Hold."""
    if swaps.empty:
        return pd.DataFrame()
    frame = swaps.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    frame["wallet"] = frame["wallet"].astype(str).str.lower()
    ciclos = []

    for wallet, group in frame.groupby("wallet"):
        inventory = deque()
        for _, row in group.sort_values("timestamp").iterrows():
            accion = row["accion"]
            pol = float(row["pol_cantidad"])
            if accion == "BUY_POL":
                inventory.append({
                    "open_time": row["timestamp"],
                    "pol_restante": pol,
                    "pol_inicial": pol,
                    "precio_open": float(row["precio_ejecutado"]),
                    "hash_open": row["hash_tx"],
                    "gas_usdc_open": float(row.get("gas_usdc", 0.0)),
                })
            elif accion == "SELL_POL" and inventory:
                sell_pol = pol
                sell_time = row["timestamp"]
                sell_price = float(row["precio_ejecutado"])
                sell_gas = float(row.get("gas_usdc", 0.0))

                while sell_pol > 1e-9 and inventory:
                    lot = inventory[0]
                    matched = min(lot["pol_restante"], sell_pol)
                    lot["pol_restante"] -= matched
                    sell_pol -= matched

                    frac_open = matched / lot["pol_inicial"] if lot["pol_inicial"] > 0 else 0.0
                    frac_close = matched / pol if pol > 0 else 0.0
                    gas_asig = lot["gas_usdc_open"] * frac_open + sell_gas * frac_close
                    pnl_bruto = matched * (sell_price - lot["precio_open"])
                    pnl_neto = pnl_bruto - gas_asig
                    costo_base = matched * lot["precio_open"]
                    retorno_neto = (pnl_neto / costo_base) if costo_base > 0 else 0.0

                    ciclos.append({
                        "wallet": wallet,
                        "open_time": lot["open_time"],
                        "close_time": sell_time,
                        "pol_cantidad": matched,
                        "precio_open": lot["precio_open"],
                        "precio_close": sell_price,
                        "pnl_neto_usdc": pnl_neto,
                        "retorno_neto": retorno_neto,
                        "gas_usdc": gas_asig,
                        "hash_open": lot["hash_open"],
                        "hash_close": row["hash_tx"],
                    })

                    if lot["pol_restante"] <= 1e-9:
                        inventory.popleft()

    res = pd.DataFrame(ciclos)
    if res.empty:
        return res
    return res.sort_values("open_time").reset_index(drop=True)


# ── 4. Ledger de Decisiones Maduras ──────────────────────────────────────────

def construir_ledger_decisiones(
    swaps_logicos: pd.DataFrame,
    ciclos_hold: pd.DataFrame,
    precios_minuto: pd.Series,
    horizontes: Mapping[str, pd.Timedelta] = HORIZONS,
    as_of: datetime | pd.Timestamp | str | None = None,
) -> pd.DataFrame:
    """Evalúa retornos y PnL forward para BUY, SELL y HOLD sin fuga temporal."""
    cutoff = _as_utc(as_of) if as_of else None
    rows = []

    # BUY y SELL
    for _, row in swaps_logicos.iterrows():
        t0 = row["timestamp"]
        p0 = float(row["precio_ejecutado"])
        pol = float(row["pol_cantidad"])
        gas = float(row.get("gas_usdc", 0.0))
        notional = float(row.get("usdc_cantidad", pol * p0))
        accion = row["accion"]

        for horiz, delta in horizontes.items():
            t1 = t0 + delta
            is_mature = cutoff is None or t1 <= cutoff
            p1 = precios_minuto.asof(t1) if is_mature and not precios_minuto.empty else np.nan

            if pd.isna(p1):
                rows.append({
                    "wallet": row["wallet"], "accion": accion, "horizonte": horiz,
                    "decision_time": t0, "forward_time": t1, "es_madura": False,
                    "precio_decision": p0, "precio_forward": np.nan,
                    "retorno_neto": np.nan, "ganancia_neta_usdc": np.nan, "gas_usdc": gas,
                })
                continue

            signo = 1.0 if accion == "BUY_POL" else -1.0
            retorno_bruto = signo * (p1 - p0) / p0
            costo_base = notional if notional > 0 else (pol * p0)
            pnl_neto = (retorno_bruto * costo_base) - gas
            retorno_neto = pnl_neto / costo_base if costo_base > 0 else 0.0

            rows.append({
                "wallet": row["wallet"], "accion": accion, "horizonte": horiz,
                "decision_time": t0, "forward_time": t1, "es_madura": True,
                "precio_decision": p0, "precio_forward": p1,
                "retorno_neto": retorno_neto, "ganancia_neta_usdc": pnl_neto, "gas_usdc": gas,
            })

    # HOLD
    if not ciclos_hold.empty:
        for _, c in ciclos_hold.iterrows():
            t0 = c["open_time"]
            t_close = c["close_time"]
            p0 = float(c["precio_open"])
            pol = float(c["pol_cantidad"])
            costo_base = pol * p0
            gas = float(c.get("gas_usdc", 0.0))

            for horiz, delta in horizontes.items():
                t1 = t0 + delta
                is_mature = (cutoff is None or t1 <= cutoff) and (t1 <= t_close)
                p1 = precios_minuto.asof(t1) if is_mature and not precios_minuto.empty else np.nan

                if pd.isna(p1):
                    rows.append({
                        "wallet": c["wallet"], "accion": "HOLD", "horizonte": horiz,
                        "decision_time": t0, "forward_time": t1, "es_madura": False,
                        "precio_decision": p0, "precio_forward": np.nan,
                        "retorno_neto": np.nan, "ganancia_neta_usdc": np.nan, "gas_usdc": gas,
                    })
                    continue

                retorno_bruto = (p1 - p0) / p0
                pnl_neto = (retorno_bruto * costo_base) - gas
                retorno_neto = pnl_neto / costo_base if costo_base > 0 else 0.0

                rows.append({
                    "wallet": c["wallet"], "accion": "HOLD", "horizonte": horiz,
                    "decision_time": t0, "forward_time": t1, "es_madura": True,
                    "precio_decision": p0, "precio_forward": p1,
                    "retorno_neto": retorno_neto, "ganancia_neta_usdc": pnl_neto, "gas_usdc": gas,
                })

    df_ledger = pd.DataFrame(rows)
    if df_ledger.empty:
        return df_ledger
    return df_ledger.sort_values("decision_time").reset_index(drop=True)


# ── 5. Métricas de Consistencia y Clasificación de Ganadoras ─────────────────

def calcular_consistency_score(tasa_acierto: float, profit_factor: float, n_decisiones: int) -> float:
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
    """Devuelve 'pending', 'insufficient', 'winner' o 'not_winner'."""
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


def resumen_estados_perfiles(perfiles: pd.DataFrame) -> dict[str, int]:
    """Cuenta estados para auditar perfiles."""
    if perfiles.empty or "winner_status" not in perfiles.columns:
        return {}
    return {str(status): int(count) for status, count in perfiles["winner_status"].value_counts().items()}


def filtrar_wallets_ganadoras(
    perfiles: pd.DataFrame,
    *,
    acciones: Iterable[str] | None = None,
    horizontes: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Filtra perfiles clasificados como 'winner' por acción y horizonte."""
    if perfiles.empty:
        return pd.DataFrame()
    winners = perfiles[perfiles["winner_status"] == "winner"].copy()
    if acciones is not None:
        winners = winners[winners["accion"].isin(tuple(acciones))]
    if horizontes is not None:
        winners = winners[winners["horizonte"].isin(tuple(horizontes))]
    order = [c for c in ("consistency_score", "pnl_neto_usdc", "n_decisiones") if c in winners.columns]
    return winners.sort_values(order, ascending=[False] * len(order)).reset_index(drop=True)


def construir_perfiles_wallet(ledger: pd.DataFrame, min_decisiones: int = 3) -> pd.DataFrame:
    """Agrupa el ledger por (wallet, acción, horizonte) y calcula métricas."""
    if ledger.empty:
        return pd.DataFrame()
    records = []
    for (wallet, accion, horizonte), g in ledger.groupby(["wallet", "accion", "horizonte"]):
        maduras = g[g["es_madura"]]
        n_decisiones = len(maduras)
        n_pendientes = len(g) - n_decisiones

        if n_decisiones == 0:
            win_rate = 0.0
            profit_factor = 0.0
            score = 0.0
            pnl_neto = 0.0
            ret_mediano = 0.0
        else:
            pnl_neto = float(maduras["ganancia_neta_usdc"].sum())
            ret_mediano = float(maduras["retorno_neto"].median())
            win_rate = float((maduras["retorno_neto"] > 0).mean())
            gains = maduras.loc[maduras["ganancia_neta_usdc"] > 0, "ganancia_neta_usdc"].sum()
            losses = abs(maduras.loc[maduras["ganancia_neta_usdc"] < 0, "ganancia_neta_usdc"].sum())
            profit_factor = 3.0 if losses == 0 and gains > 0 else (gains / losses if losses > 0 else 0.0)
            score = calcular_consistency_score(win_rate, profit_factor, n_decisiones)

        status = clasificar_winner_status(
            n_decisiones=n_decisiones,
            n_pendientes=n_pendientes,
            pnl_neto_usdc=pnl_neto,
            retorno_neto_mediano=ret_mediano,
            consistency_score=score,
            min_decisiones=min_decisiones,
        )

        records.append({
            "wallet": wallet,
            "accion": accion,
            "horizonte": horizonte,
            "n_decisiones": n_decisiones,
            "n_pendientes": n_pendientes,
            "pnl_neto_usdc": pnl_neto,
            "retorno_neto_mediano": ret_mediano,
            "tasa_acierto": win_rate,
            "profit_factor": profit_factor,
            "consistency_score": score,
            "winner_status": status,
        })

    profiles = pd.DataFrame(records)
    order = ["consistency_score", "pnl_neto_usdc", "n_decisiones"]
    return profiles.sort_values(order, ascending=[False, False, False]).reset_index(drop=True)


# ── 6. Señales Causales para Bellman ──────────────────────────────────────────

def seleccionar_wallets_dirigidas_1h(
    perfiles: pd.DataFrame,
    *,
    consistency_threshold: float = CONSISTENCY_THRESHOLD_RL,
) -> pd.DataFrame:
    """Selecciona wallets consistentes de 1h para guiar al agente Bellman."""
    if perfiles.empty:
        return pd.DataFrame()
    selected = perfiles[
        perfiles["accion"].isin(ACCIONES_RL)
        & perfiles["horizonte"].eq(HORIZONTE_WALLET_RL)
        & perfiles["winner_status"].eq("winner")
        & (pd.to_numeric(perfiles["consistency_score"], errors="coerce") >= consistency_threshold)
    ].copy()
    selected["wallet"] = selected["wallet"].astype(str).str.lower()
    selected["direccion_agente"] = selected["accion"].map(_DIRECCION_AGENTE)
    order = [c for c in ("consistency_score", "pnl_neto_usdc", "n_decisiones") if c in selected.columns]
    return selected.sort_values(order, ascending=[False] * len(order)).reset_index(drop=True)


def calcular_wallets_ganadoras_1h_consistentes(
    perfiles: pd.DataFrame,
    *,
    consistency_threshold: float = CONSISTENCY_THRESHOLD_RL,
) -> pd.DataFrame:
    """Calcula la cohorte de wallets que dirige al agente en 1 hora con score >= umbral."""
    return seleccionar_wallets_dirigidas_1h(perfiles, consistency_threshold=consistency_threshold)


def construir_perfiles_causales_1h(
    ledger: pd.DataFrame,
    *,
    as_of: datetime | pd.Timestamp | str,
    consistency_threshold: float = CONSISTENCY_THRESHOLD_RL,
    min_decisiones: int = 3,
) -> pd.DataFrame:
    """Calcula perfiles 1h conocidos estrictamente hasta as_of para evitar fuga temporal."""
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
        & (summary["consistency_score"] >= consistency_threshold)
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
    selected = seleccionar_wallets_dirigidas_1h(perfiles, consistency_threshold=consistency_threshold)
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
        "n_wallets_dirigidas": int(selected["wallet"].nunique()) if not selected.empty else 0,
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
    """Construye la señal causal horaria agregada para el estado del agente."""
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


# ── 7. Ejecutor en Memoria ───────────────────────────────────────────────────

def analizar_wallets(
    swaps: pd.DataFrame,
    min_decisiones: int = 3,
    as_of: datetime | pd.Timestamp | str | None = None,
) -> WalletAnalysisResult:
    """Ejecuta el análisis completo en memoria: swaps -> ciclos FIFO -> ledger -> perfiles -> ganadoras."""
    canonical = canonicalizar_swaps_logicos(swaps)
    precios = construir_precios_por_minuto_desde_swaps(canonical)
    ciclos = construir_ciclos_hold(canonical)
    ledger = construir_ledger_decisiones(canonical, ciclos, precios, as_of=as_of)
    perfiles = construir_perfiles_wallet(ledger, min_decisiones=min_decisiones)
    ganadoras = filtrar_wallets_ganadoras(perfiles)
    estados = resumen_estados_perfiles(perfiles)

    return WalletAnalysisResult(
        perfiles=perfiles,
        ganadoras=ganadoras,
        ledger=ledger,
        swaps_logicos=canonical,
        ciclos_hold=ciclos,
        estados=estados,
    )


__all__ = [
    "CONSISTENCY_THRESHOLD_RL",
    "DEFAULT_LOOKBACK_HOURS",
    "DEFAULT_POOL_ADDRESS",
    "HORIZONS",
    "MIN_CONSISTENCY_SCORE",
    "MIN_DECISIONES_GANADORA",
    "WalletAnalysisResult",
    "WalletView",
    "analizar_wallets",
    "calcular_consenso_wallets_1h",
    "calcular_consistency_score",
    "calcular_wallets_ganadoras_1h_consistentes",
    "canonicalizar_swaps_logicos",
    "clasificar_winner_status",
    "construir_ciclos_hold",
    "construir_ledger_decisiones",
    "construir_perfiles_causales_1h",
    "construir_perfiles_wallet",
    "construir_precios_por_minuto_desde_swaps",
    "construir_senales_wallets_horarias",
    "filtrar_wallets_ganadoras",
    "resumen_estados_perfiles",
    "seleccionar_wallets_dirigidas_1h",
]
