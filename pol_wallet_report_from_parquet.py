"""Informe local de wallets POL a partir de un Parquet ya descargado.

Este archivo no se conecta a Alchemy, no consulta Yahoo Finance y no necesita
una clave API. Sirve para reutilizar un ``swaps_*.parquet`` que ya esté en
Colab o en el computador y producir un informe visual de actividad, Buy/Sell y
ciclos Hold. Por tanto es inmediato incluso si la descarga on-chain anterior
tardó una hora.

Para etiquetar Buy/Sell/Hold, genera el precio horario POL/USDC con los swaps
del propio Parquet. Las últimas 1 h, 4 h y 1 d permanecen pendientes cuando el
archivo todavía no contiene el cierre posterior necesario.
"""

from __future__ import annotations

import hashlib
import json
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pol_wallet_winners import (
    calcular_consistency_score,
    clasificar_winner_status,
    filtrar_wallets_ganadoras,
    resumen_estados_perfiles,
)


HORIZONS = {
    "1h": pd.Timedelta(hours=1),
    "4h": pd.Timedelta(hours=4),
    "1d": pd.Timedelta(days=1),
}


def _as_utc(value: datetime | pd.Timestamp | str | None) -> pd.Timestamp:
    stamp = pd.Timestamp.now(tz="UTC") if value is None else pd.Timestamp(value)
    return stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")


def _short_wallet(value: object) -> str:
    wallet = str(value)
    return f"{wallet[:6]}…{wallet[-4:]}" if len(wallet) > 14 else wallet


def _empty_logical_swaps() -> pd.DataFrame:
    return pd.DataFrame(columns=[
        "timestamp", "block_number", "hash_tx", "wallet", "direccion", "pol_delta",
        "usdc_delta", "pol_cantidad", "notional_usdc", "precio_ejecutado", "gas_usdc",
        "event_count",
    ])


def canonicalizar_swaps_logicos(swaps: pd.DataFrame) -> pd.DataFrame:
    """Convierte eventos o el Parquet antiguo ``Buy/Sell`` en swaps lógicos.

    Una transacción de una wallet se agrega en una sola fila. Esto evita contar
    varias veces el gas de rutas multihop y descarta flujos POL netos cero de
    los cálculos de Buy/Sell.
    """
    if swaps.empty:
        return _empty_logical_swaps()

    frame = swaps.copy()
    rename = {
        "bloque": "block_number",
        "transaction_hash": "hash_tx",
        "tx_hash": "hash_tx",
        "from": "wallet",
    }
    frame = frame.rename(columns={old: new for old, new in rename.items() if old in frame.columns and new not in frame.columns})
    if "block_number" not in frame.columns:
        frame["block_number"] = pd.NA
    if "log_index" not in frame.columns:
        frame["log_index"] = 0
    if "gas_usdc" not in frame.columns:
        frame["gas_usdc"] = 0.0
    if "pol_cantidad" not in frame.columns and "pol_amount" in frame.columns:
        frame["pol_cantidad"] = frame["pol_amount"]
    if "usdc_cantidad" not in frame.columns and "usdc_amount" in frame.columns:
        frame["usdc_cantidad"] = frame["usdc_amount"]

    if "pol_delta" not in frame.columns and {"direccion", "pol_cantidad"}.issubset(frame.columns):
        direction = frame["direccion"].astype(str).str.upper()
        quantity = pd.to_numeric(frame["pol_cantidad"], errors="coerce").fillna(0.0)
        frame["pol_delta"] = np.select(
            [direction.isin(["BUY", "BUY_POL"]), direction.isin(["SELL", "SELL_POL"])],
            [quantity, -quantity], default=0.0,
        )
    if "usdc_delta" not in frame.columns and {"direccion", "usdc_cantidad"}.issubset(frame.columns):
        direction = frame["direccion"].astype(str).str.upper()
        quantity = pd.to_numeric(frame["usdc_cantidad"], errors="coerce").fillna(0.0)
        frame["usdc_delta"] = np.select(
            [direction.isin(["BUY", "BUY_POL"]), direction.isin(["SELL", "SELL_POL"])],
            [-quantity, quantity], default=0.0,
        )

    required = {"timestamp", "hash_tx", "wallet", "pol_delta", "usdc_delta"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(
            "El Parquet no tiene las columnas necesarias: " + ", ".join(sorted(missing))
        )

    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    frame = frame.dropna(subset=["timestamp", "hash_tx", "wallet"]).copy()
    frame["hash_tx"] = frame["hash_tx"].astype(str).str.lower()
    frame["wallet"] = frame["wallet"].astype(str).str.lower()
    for column in ("pol_delta", "usdc_delta", "gas_usdc", "log_index", "block_number"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0.0)

    grouped = frame.groupby(["wallet", "hash_tx"], as_index=False).agg(
        timestamp=("timestamp", "min"),
        block_number=("block_number", "min"),
        pol_delta=("pol_delta", "sum"),
        usdc_delta=("usdc_delta", "sum"),
        gas_usdc=("gas_usdc", "max"),
        event_count=("log_index", "count"),
    )
    grouped["pol_cantidad"] = grouped["pol_delta"].abs()
    grouped["notional_usdc"] = grouped["usdc_delta"].abs()
    grouped["precio_ejecutado"] = np.where(
        grouped["pol_cantidad"] > 0,
        grouped["notional_usdc"] / grouped["pol_cantidad"],
        np.nan,
    )
    grouped["direccion"] = np.select(
        [grouped["pol_delta"] > 0, grouped["pol_delta"] < 0],
        ["BUY_POL", "SELL_POL"], default="AMBIGUOUS",
    )
    return grouped.sort_values(["timestamp", "hash_tx"]).reset_index(drop=True)[_empty_logical_swaps().columns]


def construir_ciclos_hold(swaps_logicos: pd.DataFrame) -> pd.DataFrame:
    """Hace matching FIFO de compras y ventas para medir Hold sin inventarlo."""
    columns = [
        "cycle_id", "wallet", "buy_hash_tx", "sell_hash_tx", "opened_at", "closed_at",
        "pol_cantidad", "entry_price", "exit_price", "entry_gas_usdc", "exit_gas_usdc",
        "duration_hours", "pnl_realizado_usdc", "cycle_status",
    ]
    if swaps_logicos.empty:
        return pd.DataFrame(columns=columns)

    trades = swaps_logicos[swaps_logicos["direccion"].isin(["BUY_POL", "SELL_POL"])].copy()
    trades["timestamp"] = pd.to_datetime(trades["timestamp"], utc=True)
    trades = trades.sort_values(["wallet", "timestamp", "hash_tx"])
    cycles: list[dict[str, Any]] = []
    sequence = 0
    for wallet, wallet_trades in trades.groupby("wallet", sort=False):
        open_lots: deque[dict[str, Any]] = deque()
        for _, trade in wallet_trades.iterrows():
            quantity = float(trade["pol_cantidad"])
            if quantity <= 0:
                continue
            price = float(trade["precio_ejecutado"])
            gas_per_pol = float(trade["gas_usdc"] or 0.0) / quantity
            if trade["direccion"] == "BUY_POL":
                open_lots.append({
                    "remaining": quantity, "opened_at": trade["timestamp"], "hash_tx": trade["hash_tx"],
                    "price": price, "gas_per_pol": gas_per_pol,
                })
                continue

            remaining = quantity
            while remaining > 1e-12 and open_lots:
                lot = open_lots[0]
                matched = min(remaining, lot["remaining"])
                sequence += 1
                entry_gas = matched * lot["gas_per_pol"]
                exit_gas = matched * gas_per_pol
                cycles.append({
                    "cycle_id": f"{wallet}:{sequence}", "wallet": wallet,
                    "buy_hash_tx": lot["hash_tx"], "sell_hash_tx": trade["hash_tx"],
                    "opened_at": lot["opened_at"], "closed_at": trade["timestamp"],
                    "pol_cantidad": matched, "entry_price": lot["price"], "exit_price": price,
                    "entry_gas_usdc": entry_gas, "exit_gas_usdc": exit_gas,
                    "duration_hours": (trade["timestamp"] - lot["opened_at"]).total_seconds() / 3600,
                    "pnl_realizado_usdc": matched * (price - lot["price"]) - entry_gas - exit_gas,
                    "cycle_status": "CLOSED",
                })
                lot["remaining"] -= matched
                remaining -= matched
                if lot["remaining"] <= 1e-12:
                    open_lots.popleft()
            if remaining > 1e-12:
                sequence += 1
                cycles.append({
                    "cycle_id": f"{wallet}:{sequence}", "wallet": wallet,
                    "buy_hash_tx": None, "sell_hash_tx": trade["hash_tx"],
                    "opened_at": pd.NaT, "closed_at": trade["timestamp"], "pol_cantidad": remaining,
                    "entry_price": np.nan, "exit_price": price, "entry_gas_usdc": np.nan,
                    "exit_gas_usdc": remaining * gas_per_pol, "duration_hours": np.nan,
                    "pnl_realizado_usdc": np.nan, "cycle_status": "CENSORED_PREEXISTING",
                })
        for lot in open_lots:
            sequence += 1
            cycles.append({
                "cycle_id": f"{wallet}:{sequence}", "wallet": wallet,
                "buy_hash_tx": lot["hash_tx"], "sell_hash_tx": None,
                "opened_at": lot["opened_at"], "closed_at": pd.NaT,
                "pol_cantidad": lot["remaining"], "entry_price": lot["price"], "exit_price": np.nan,
                "entry_gas_usdc": lot["remaining"] * lot["gas_per_pol"], "exit_gas_usdc": np.nan,
                "duration_hours": np.nan, "pnl_realizado_usdc": np.nan, "cycle_status": "CENSORED_OPEN",
            })
    return pd.DataFrame(cycles, columns=columns).sort_values(
        ["wallet", "opened_at", "closed_at"], na_position="last"
    ).reset_index(drop=True)


def construir_precios_horarios_desde_swaps(swaps_logicos: pd.DataFrame) -> pd.DataFrame:
    """Construye el cierre horario POL/USDC desde los swaps ya descargados.

    Es una alternativa local a Yahoo Finance: el cierre de cada hora es el
    último precio ejecutado observado en el pool. Se rellenan horas sin swap
    con el último cierre conocido, nunca con un precio futuro.
    """
    if swaps_logicos.empty:
        return pd.DataFrame(columns=["Close", "source"])
    frame = swaps_logicos.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    frame["precio_ejecutado"] = pd.to_numeric(frame["precio_ejecutado"], errors="coerce")
    frame = frame[(frame["precio_ejecutado"] > 0) & frame["direccion"].isin(["BUY_POL", "SELL_POL"])]
    if frame.empty:
        return pd.DataFrame(columns=["Close", "source"])
    frame = frame.sort_values(["timestamp", "hash_tx"]).copy()
    frame["hour"] = frame["timestamp"].dt.floor("h")
    close = frame.groupby("hour", sort=True)["precio_ejecutado"].last().rename("Close")
    close = close.resample("h").last().ffill()
    result = close.to_frame()
    result["source"] = "POOL_SWAP_CLOSE"
    return result


def _normalizar_precios(precios: pd.DataFrame) -> pd.Series:
    if precios.empty:
        return pd.Series(dtype=float, name="price")
    frame = precios.copy()
    if "timestamp" in frame.columns:
        index = pd.to_datetime(frame.pop("timestamp"), utc=True)
    else:
        index = pd.to_datetime(frame.index, utc=True)
    column = "Close" if "Close" in frame.columns else "price"
    if column not in frame.columns:
        raise ValueError("La tabla de precios debe contener Close o price.")
    values = pd.to_numeric(frame[column], errors="coerce")
    series = pd.Series(values.to_numpy(), index=index, name="price").dropna()
    return series.groupby(series.index.floor("h")).last().sort_index()


def _etiquetar_decision(
    *,
    action: str,
    decision_price: float,
    notional_usdc: float,
    gas_usdc: float,
    decision_time: pd.Timestamp,
    horizon: str,
    prices: pd.Series,
    as_of: pd.Timestamp,
) -> dict[str, Any]:
    forward_time = decision_time.floor("h") + HORIZONS[horizon]
    base = {
        "forward_time": forward_time, "forward_price": np.nan,
        "retorno_bruto": np.nan, "retorno_neto": np.nan, "ganancia_neta_usdc": np.nan,
    }
    if forward_time > as_of.floor("h"):
        return {**base, "maturity_status": "PENDING"}
    if forward_time not in prices.index or not pd.notna(decision_price) or decision_price <= 0:
        return {**base, "maturity_status": "MISSING_PRICE"}
    forward_price = float(prices.loc[forward_time])
    gross = (
        (forward_price - decision_price) / decision_price
        if action in {"BUY_POL", "HOLD"}
        else (decision_price - forward_price) / decision_price
    )
    commission = 0.0 if action == "HOLD" else gas_usdc / notional_usdc if notional_usdc > 0 else np.nan
    net = gross - commission
    return {
        **base, "forward_price": forward_price, "retorno_bruto": gross, "retorno_neto": net,
        "ganancia_neta_usdc": net * notional_usdc, "maturity_status": "COMPLETED",
    }


def construir_ledger_decisiones(
    swaps_logicos: pd.DataFrame,
    ciclos_hold: pd.DataFrame,
    precios: pd.DataFrame,
    *,
    as_of: datetime | pd.Timestamp | str,
) -> pd.DataFrame:
    """Calcula Buy/Sell/Hold forward con cierres del pool, sin llamadas externas."""
    prices = _normalizar_precios(precios)
    cutoff = _as_utc(as_of)
    records: list[dict[str, Any]] = []
    trades = swaps_logicos[swaps_logicos["direccion"].isin(["BUY_POL", "SELL_POL"])]
    for _, trade in trades.iterrows():
        decision_time = _as_utc(trade["timestamp"])
        for horizon in HORIZONS:
            record = {
                "decision_id": f"trade:{trade['wallet']}:{trade['hash_tx']}:{horizon}",
                "wallet": trade["wallet"], "accion": trade["direccion"], "horizonte": horizon,
                "source_type": "TRADE", "source_id": trade["hash_tx"], "decision_time": decision_time,
                "decision_price": float(trade["precio_ejecutado"]),
                "notional_usdc": float(trade["notional_usdc"]), "gas_usdc": float(trade["gas_usdc"]),
            }
            records.append({**record, **_etiquetar_decision(
                action=record["accion"], decision_price=record["decision_price"],
                notional_usdc=record["notional_usdc"], gas_usdc=record["gas_usdc"],
                decision_time=decision_time, horizon=horizon, prices=prices, as_of=cutoff,
            )})
    closed = ciclos_hold[ciclos_hold["cycle_status"] == "CLOSED"] if not ciclos_hold.empty else ciclos_hold
    for _, cycle in closed.iterrows():
        decision_time = _as_utc(cycle["opened_at"])
        for horizon, delta in HORIZONS.items():
            if float(cycle["duration_hours"]) + 1e-12 < delta.total_seconds() / 3600:
                continue
            notional = float(cycle["pol_cantidad"] * cycle["entry_price"])
            record = {
                "decision_id": f"hold:{cycle['cycle_id']}:{horizon}",
                "wallet": cycle["wallet"], "accion": "HOLD", "horizonte": horizon,
                "source_type": "HOLD_CYCLE", "source_id": cycle["cycle_id"], "decision_time": decision_time,
                "decision_price": float(cycle["entry_price"]), "notional_usdc": notional, "gas_usdc": 0.0,
            }
            records.append({**record, **_etiquetar_decision(
                action="HOLD", decision_price=record["decision_price"], notional_usdc=notional,
                gas_usdc=0.0, decision_time=decision_time, horizon=horizon, prices=prices, as_of=cutoff,
            )})
    columns = [
        "decision_id", "wallet", "accion", "horizonte", "source_type", "source_id", "decision_time",
        "decision_price", "notional_usdc", "gas_usdc", "forward_time", "forward_price",
        "retorno_bruto", "retorno_neto", "ganancia_neta_usdc", "maturity_status",
    ]
    return pd.DataFrame(records, columns=columns).sort_values(
        ["decision_time", "wallet", "decision_id"]
    ).reset_index(drop=True) if records else pd.DataFrame(columns=columns)


def _profit_factor(values: pd.Series) -> float:
    positives = float(values.clip(lower=0).sum())
    negatives = float((-values.clip(upper=0)).sum())
    return 3.0 if negatives == 0 and positives > 0 else 0.0 if negatives == 0 else min(positives / negatives, 3.0)


def construir_perfiles_wallet(
    ledger: pd.DataFrame,
    swaps_logicos: pd.DataFrame,
    ciclos_hold: pd.DataFrame,
    *,
    min_decisiones: int = 3,
) -> pd.DataFrame:
    """Clasifica perfiles con la regla de ganadora definida para el proyecto."""
    columns = [
        "wallet", "accion", "horizonte", "n_decisiones", "n_pendientes", "pnl_neto_usdc",
        "retorno_neto_mediano", "tasa_acierto", "profit_factor", "consistency_score",
        "winner_status", "perfil_conductual", "duration_hold_mediana_horas",
    ]
    if ledger.empty:
        return pd.DataFrame(columns=columns)
    closed = ciclos_hold[ciclos_hold["cycle_status"] == "CLOSED"] if not ciclos_hold.empty else ciclos_hold
    durations = closed.groupby("wallet")["duration_hours"].median() if not closed.empty else pd.Series(dtype=float)
    active = swaps_logicos[swaps_logicos["direccion"].isin(["BUY_POL", "SELL_POL"])]
    flow = active.groupby("wallet")["pol_delta"].sum() if not active.empty else pd.Series(dtype=float)
    records: list[dict[str, Any]] = []
    for (wallet, action, horizon), group in ledger.groupby(["wallet", "accion", "horizonte"]):
        completed = group[group["maturity_status"] == "COMPLETED"]
        decisions = len(completed)
        pending = int((group["maturity_status"] == "PENDING").sum())
        if decisions:
            pnl = float(completed["ganancia_neta_usdc"].sum())
            median_return = float(completed["retorno_neto"].median())
            hit_rate = float((completed["retorno_neto"] > 0).mean())
            profit_factor = _profit_factor(completed["ganancia_neta_usdc"])
            score = calcular_consistency_score(hit_rate, profit_factor, decisions)
        else:
            pnl = median_return = hit_rate = profit_factor = score = np.nan
        status = clasificar_winner_status(
            n_decisiones=decisions,
            n_pendientes=pending,
            pnl_neto_usdc=pnl,
            retorno_neto_mediano=median_return,
            consistency_score=score,
            min_decisiones=min_decisiones,
        )
        net_flow = float(flow.get(wallet, 0.0))
        bias = "NET_BUYER" if net_flow > 0 else "NET_SELLER" if net_flow < 0 else "BALANCED"
        duration = float(durations.get(wallet, np.nan))
        tempo = "UNKNOWN_HOLD" if pd.isna(duration) else "SCALPER" if duration < 1 else "SWING" if duration < 24 else "LONG_HOLDER"
        records.append({
            "wallet": wallet, "accion": action, "horizonte": horizon,
            "n_decisiones": decisions, "n_pendientes": pending, "pnl_neto_usdc": pnl,
            "retorno_neto_mediano": median_return, "tasa_acierto": hit_rate,
            "profit_factor": profit_factor, "consistency_score": score, "winner_status": status,
            "perfil_conductual": f"{bias}_{tempo}", "duration_hold_mediana_horas": duration,
        })
    return pd.DataFrame(records, columns=columns).sort_values(
        ["winner_status", "consistency_score"], ascending=[True, False], na_position="last"
    ).reset_index(drop=True)


def construir_resumen_wallets(swaps_logicos: pd.DataFrame, ciclos_hold: pd.DataFrame) -> pd.DataFrame:
    """Resume rasgos observables sin usar un precio futuro ni etiquetar ganadores."""
    columns = [
        "wallet", "n_swaps", "n_buy", "n_sell", "notional_usdc", "flujo_neto_pol",
        "ratio_buy_sell", "n_ciclos_cerrados", "pnl_realizado_usdc",
        "duration_hold_media_horas", "duration_hold_mediana_horas", "perfil_conductual",
    ]
    trades = swaps_logicos[swaps_logicos["direccion"].isin(["BUY_POL", "SELL_POL"])].copy()
    if trades.empty:
        return pd.DataFrame(columns=columns)
    summary = trades.groupby("wallet", as_index=False).agg(
        n_swaps=("hash_tx", "count"),
        n_buy=("direccion", lambda values: int((values == "BUY_POL").sum())),
        n_sell=("direccion", lambda values: int((values == "SELL_POL").sum())),
        notional_usdc=("notional_usdc", "sum"),
        flujo_neto_pol=("pol_delta", "sum"),
    )
    summary["ratio_buy_sell"] = summary["n_buy"] / summary["n_sell"].clip(lower=1)

    closed = ciclos_hold[ciclos_hold["cycle_status"] == "CLOSED"].copy()
    if closed.empty:
        summary["n_ciclos_cerrados"] = 0
        summary["pnl_realizado_usdc"] = np.nan
        summary["duration_hold_media_horas"] = np.nan
        summary["duration_hold_mediana_horas"] = np.nan
    else:
        holds = closed.groupby("wallet", as_index=False).agg(
            n_ciclos_cerrados=("cycle_id", "count"),
            pnl_realizado_usdc=("pnl_realizado_usdc", "sum"),
            duration_hold_media_horas=("duration_hours", "mean"),
            duration_hold_mediana_horas=("duration_hours", "median"),
        )
        summary = summary.merge(holds, on="wallet", how="left")
        summary["n_ciclos_cerrados"] = summary["n_ciclos_cerrados"].fillna(0).astype(int)

    flow = np.select(
        [summary["flujo_neto_pol"] > 0, summary["flujo_neto_pol"] < 0],
        ["NET_BUYER", "NET_SELLER"], default="BALANCED",
    )
    duration = summary["duration_hold_mediana_horas"]
    tempo = np.select(
        [duration.isna(), duration < 1, duration < 24],
        ["UNKNOWN_HOLD", "SCALPER", "SWING"], default="LONG_HOLDER",
    )
    summary["perfil_conductual"] = pd.Series(flow, index=summary.index) + "_" + pd.Series(tempo, index=summary.index)
    return summary.sort_values("n_swaps", ascending=False).reset_index(drop=True)[columns]


def _top_bottom(
    frame: pd.DataFrame, metric: str, n: int = 5, unique_columns: list[str] | None = None,
) -> pd.DataFrame:
    source = frame.dropna(subset=[metric]).copy()
    if source.empty:
        return source.assign(grupo_extremo=pd.Series(dtype=str))
    top = source.nlargest(n, metric).sort_values(metric, ascending=False).assign(grupo_extremo="Top 5 · descendente")
    bottom = source.nsmallest(n, metric).sort_values(metric, ascending=True).assign(grupo_extremo="Bottom 5 · ascendente")
    unique_columns = unique_columns or ["wallet"]
    return pd.concat([top, bottom]).drop_duplicates(unique_columns, keep="first")


def crear_figuras(
    resumen: pd.DataFrame,
    swaps_logicos: pd.DataFrame,
    ciclos_hold: pd.DataFrame,
    perfiles: pd.DataFrame,
) -> dict[str, Any]:
    """Crea vistas de ganadoras y sus rasgos, limitadas a top/bottom 5."""
    try:
        import plotly.express as px
        import plotly.graph_objects as go
    except ImportError as exc:  # pragma: no cover - depende del entorno.
        raise RuntimeError("Instala plotly: pip install plotly") from exc

    def empty(title: str, detail: str) -> Any:
        figure = go.Figure()
        figure.add_annotation(text=detail, showarrow=False, font={"size": 17}, x=0.5, y=0.5)
        figure.update_layout(title=title, template="plotly_white")
        return figure

    winner_profiles = filtrar_wallets_ganadoras(perfiles) if not perfiles.empty else perfiles
    winner_wallets = set(winner_profiles.get("wallet", pd.Series(dtype=str)))
    focus = resumen[resumen["wallet"].isin(winner_wallets)].copy() if winner_wallets else resumen.copy()
    cohort_note = (
        "Cohorte: wallets ganadoras (mínimo 3 decisiones maduras)."
        if winner_wallets else
        "No hay wallets ganadoras todavía; se muestran todas como diagnóstico."
    )

    def decorate(figure: Any) -> Any:
        figure.update_layout(template="plotly_white", legend_title_text="", margin={"t": 90, "l": 70, "r": 30, "b": 60})
        figure.add_annotation(
            text="Precio forward: cierre horario derivado de swaps del propio Parquet. " + cohort_note,
            x=0.5, y=1.12, xref="paper", yref="paper", showarrow=False, font={"color": "#475569"},
        )
        return figure

    if winner_profiles.empty:
        statuses = resumen_estados_perfiles(perfiles)
        detail = "No hay perfiles ganadores. Estados: " + ", ".join(
            f"{key}={value}" for key, value in statuses.items()
        ) if statuses else "No se pudieron formar perfiles con el Parquet."
        ranking = empty("Ranking de wallets ganadoras", detail)
    else:
        ranked = _top_bottom(
            winner_profiles, "consistency_score", unique_columns=["wallet", "accion", "horizonte"],
        ).copy()
        ranked["perfil"] = (
            ranked["wallet"].map(_short_wallet) + " · "
            + ranked["accion"].str.replace("_POL", "", regex=False) + " · " + ranked["horizonte"]
        )
        ranking = px.bar(
            ranked, x="consistency_score", y="perfil", color="accion", orientation="h",
            facet_col="grupo_extremo", hover_data=["wallet", "pnl_neto_usdc", "n_decisiones", "tasa_acierto"],
            title="Wallets ganadoras: Top 5 y Bottom 5 por consistencia",
            labels={"consistency_score": "Score de consistencia", "perfil": "Wallet · acción · horizonte"},
        )
        ranking.for_each_yaxis(lambda axis: axis.update(autorange="reversed"))
        decorate(ranking)

    activity_by_action = swaps_logicos[swaps_logicos["direccion"].isin(["BUY_POL", "SELL_POL"])].groupby(
        ["wallet", "direccion"], as_index=False
    ).agg(n_swaps=("hash_tx", "count"), notional_usdc=("notional_usdc", "sum")) if not swaps_logicos.empty else pd.DataFrame()

    selected_activity = _top_bottom(focus, "n_swaps")
    if selected_activity.empty or activity_by_action.empty:
        activity = empty("Actividad Buy/Sell", "No hay swaps Buy/Sell que mostrar")
    else:
        data = activity_by_action.merge(selected_activity[["wallet", "grupo_extremo"]], on="wallet", how="inner")
        data["wallet_corta"] = data["wallet"].map(_short_wallet)
        activity = px.bar(
            data, x="n_swaps", y="wallet_corta", color="direccion", orientation="h", barmode="stack",
            facet_col="grupo_extremo", hover_data=["wallet", "notional_usdc"],
            title="Comportamiento Buy/Sell: Top 5 y Bottom 5 por actividad",
            labels={"n_swaps": "Swaps observados", "wallet_corta": "Wallet"},
        )
        activity.for_each_yaxis(lambda axis: axis.update(autorange="reversed"))
        decorate(activity)

    selected_volume = _top_bottom(focus, "notional_usdc")
    if selected_volume.empty or activity_by_action.empty:
        volume = empty("Volumen Buy/Sell", "No hay volumen que mostrar")
    else:
        data = activity_by_action.merge(selected_volume[["wallet", "grupo_extremo"]], on="wallet", how="inner")
        data["wallet_corta"] = data["wallet"].map(_short_wallet)
        volume = px.bar(
            data, x="notional_usdc", y="wallet_corta", color="direccion", orientation="h", barmode="stack",
            facet_col="grupo_extremo", hover_data=["wallet", "n_swaps"],
            title="Comportamiento Buy/Sell: Top 5 y Bottom 5 por volumen",
            labels={"notional_usdc": "Nocional USDC", "wallet_corta": "Wallet"},
        )
        volume.for_each_yaxis(lambda axis: axis.update(autorange="reversed"))
        decorate(volume)

    selected_duration = _top_bottom(focus[focus["n_ciclos_cerrados"] > 0], "duration_hold_mediana_horas")
    if selected_duration.empty:
        hold_duration = empty("Duración Hold", "No hubo ciclos completos compra → venta en el Parquet")
    else:
        data = selected_duration.copy()
        data["wallet_corta"] = data["wallet"].map(_short_wallet)
        hold_duration = px.bar(
            data, x="duration_hold_mediana_horas", y="wallet_corta", color="perfil_conductual", orientation="h",
            facet_col="grupo_extremo", hover_data=["wallet", "n_ciclos_cerrados", "pnl_realizado_usdc"],
            title="Hold: Top 5 y Bottom 5 por duración mediana",
            labels={"duration_hold_mediana_horas": "Horas de Hold", "wallet_corta": "Wallet"},
        )
        hold_duration.for_each_yaxis(lambda axis: axis.update(autorange="reversed"))
        decorate(hold_duration)

    matrix_source = selected_activity.copy()
    if matrix_source.empty:
        matrix = empty("Matriz de rasgos", "No hay wallets para resumir")
    else:
        matrix_source["BUY_POL"] = matrix_source["n_buy"]
        matrix_source["SELL_POL"] = matrix_source["n_sell"]
        matrix_source["HOLD_CLOSED"] = matrix_source["n_ciclos_cerrados"]
        matrix_source["wallet_corta"] = matrix_source["wallet"].map(_short_wallet)
        matrix = go.Figure(go.Heatmap(
            z=matrix_source[["BUY_POL", "SELL_POL", "HOLD_CLOSED"]].to_numpy(),
            x=["Compras POL", "Ventas POL", "Ciclos Hold cerrados"],
            y=matrix_source["wallet_corta"].tolist(), colorscale="Blues",
            text=matrix_source[["BUY_POL", "SELL_POL", "HOLD_CLOSED"]].to_numpy(), texttemplate="%{text}",
            colorbar_title="Conteo",
        ))
        matrix.update_layout(title="Matriz de rasgos de las wallets seleccionadas")
        decorate(matrix)

    closed = ciclos_hold[ciclos_hold["cycle_status"] == "CLOSED"].copy() if not ciclos_hold.empty else ciclos_hold
    if closed.empty:
        timeline = empty("Ciclos Hold", "No hubo ciclos completos compra → venta para dibujar")
    else:
        selected_pnl = _top_bottom(focus[focus["n_ciclos_cerrados"] > 0], "pnl_realizado_usdc")
        data = closed[closed["wallet"].isin(selected_pnl["wallet"])].copy()
        data["wallet_corta"] = data["wallet"].map(_short_wallet)
        timeline = px.timeline(
            data, x_start="opened_at", x_end="closed_at", y="wallet_corta", color="pnl_realizado_usdc",
            hover_data=["pol_cantidad", "duration_hours", "entry_price", "exit_price"],
            title="Ciclos Hold: wallets con mayor y menor PnL realizado",
            labels={"wallet_corta": "Wallet", "pnl_realizado_usdc": "PnL realizado (USDC)"},
            color_continuous_scale="RdYlGn",
        )
        timeline.update_yaxes(autorange="reversed")
        decorate(timeline)

    return {
        "ranking_ganadoras": ranking,
        "actividad_buy_sell": activity,
        "volumen_buy_sell": volume,
        "duracion_hold": hold_duration,
        "matriz_rasgos": matrix,
        "ciclos_hold": timeline,
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _new_snapshot_dir(output_dir: str | Path, as_of: pd.Timestamp) -> Path:
    parent = Path(output_dir)
    base = parent / f"snapshot_{as_of.strftime('%Y%m%dT%H%M%SZ')}"
    candidate = base
    suffix = 2
    while candidate.exists():
        candidate = parent / f"{base.name}_{suffix:02d}"
        suffix += 1
    candidate.mkdir(parents=True, exist_ok=False)
    return candidate


def _resolve_parquet(input_parquet: str | Path) -> Path:
    path = Path(input_parquet)
    if path.is_file() and path.suffix.lower() == ".parquet":
        return path
    if path.is_dir():
        candidates = sorted(path.glob("swaps_*.parquet")) or sorted(path.glob("*.parquet"))
        if candidates:
            return candidates[-1]
    raise FileNotFoundError(
        f"No encontré un Parquet en {path}. Indica el archivo swaps_...parquet o su carpeta."
    )


def generar_informe_desde_parquet(
    *,
    parquet_path: str | Path,
    output_dir: str | Path = "data/derived/parquet_wallet_report",
    min_decisiones: int = 3,
    export_png: bool = False,
) -> Path:
    """Genera el informe sin hacer ninguna llamada de red.

    ``parquet_path`` puede ser el archivo descargado o una carpeta que contenga
    un único ``swaps_*.parquet``. El resultado contiene perfiles Buy/Sell/Hold
    y un informe HTML. No hace ninguna llamada de red.
    """
    source = _resolve_parquet(parquet_path)
    raw = pd.read_parquet(source)
    logical = canonicalizar_swaps_logicos(raw)
    cycles = construir_ciclos_hold(logical)
    summary = construir_resumen_wallets(logical, cycles)
    as_of = logical["timestamp"].max() if not logical.empty else pd.Timestamp.now(tz="UTC")
    prices = construir_precios_horarios_desde_swaps(logical)
    ledger = construir_ledger_decisiones(logical, cycles, prices, as_of=as_of)
    profiles = construir_perfiles_wallet(
        ledger, logical, cycles, min_decisiones=min_decisiones,
    )
    root = _new_snapshot_dir(output_dir, _as_utc(as_of))

    logical.to_parquet(root / "swaps_logicos.parquet", index=False)
    cycles.to_parquet(root / "ciclos_hold.parquet", index=False)
    summary.to_parquet(root / "resumen_wallets.parquet", index=False)
    prices.reset_index(names="timestamp").to_parquet(root / "precios_horarios_pool.parquet", index=False)
    ledger.to_parquet(root / "ledger_decisiones.parquet", index=False)
    profiles.to_parquet(root / "perfiles_wallet.parquet", index=False)

    figures = crear_figuras(summary, logical, cycles, profiles)
    winner_count = int((profiles["winner_status"] == "winner").sum()) if not profiles.empty else 0
    sections = [
        "<h1>Informe local de wallets POL</h1>",
        "<p>Fuente: Parquet ya descargado. Este informe no consulta Alchemy ni Yahoo Finance. "
        "El precio forward es el cierre horario de los swaps del propio pool. "
        f"Perfiles ganadores encontrados: <strong>{winner_count}</strong>.</p>",
    ]
    png_errors: list[dict[str, str]] = []
    for index, (name, figure) in enumerate(figures.items()):
        figure.write_html(root / f"{name}.html", include_plotlyjs="cdn")
        sections.append(figure.to_html(full_html=False, include_plotlyjs="cdn" if index == 0 else False))
        if export_png:
            try:
                figure.write_image(root / f"{name}.png", scale=2)
            except (ImportError, OSError, ValueError) as exc:
                png_errors.append({"figure": name, "error": str(exc)})
                print(f"[gráfica] no se pudo exportar {name}.png: {exc}")
    (root / "informe_wallets.html").write_text(
        "<html><head><meta charset='utf-8'><title>Informe wallets POL</title></head><body>"
        + "\n".join(sections) + "</body></html>", encoding="utf-8",
    )
    manifest = {
        "schema_version": 1,
        "source": "Parquet local (sin llamadas de red)",
        "price_source": "Cierre horario del pool derivado de precio_ejecutado",
        "source_parquet": {"path": str(source), "sha256": _sha256_file(source), "rows": int(len(raw))},
        "as_of_utc": _as_utc(as_of).isoformat(),
        "limitations": [
            "El precio forward procede de cierres del pool, no de una fuente de mercado independiente.",
            "Las últimas observaciones quedan PENDING si no hay cierre posterior en el Parquet.",
            "HOLD sólo representa ciclos FIFO compra->venta completamente observados.",
        ],
        "winner_definition": {
            "min_decisiones_maduras": min_decisiones,
            "pnl_neto_positivo": True,
            "retorno_neto_mediano_positivo": True,
            "consistency_score_minimo": 0.60,
        },
        "rows": {
            "swaps_logicos": int(len(logical)), "ciclos_hold": int(len(cycles)),
            "ledger_decisiones": int(len(ledger)), "perfiles_wallet": int(len(profiles)),
            "resumen_wallets": int(len(summary)),
        },
        "png_export_errors": png_errors,
    }
    (root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Informe creado sin red: {root}")
    return root


if __name__ == "__main__":
    print(
        "Módulo cargado. En Colab usa generar_informe_desde_parquet("
        "parquet_path='/content/.../swaps_....parquet')."
    )
