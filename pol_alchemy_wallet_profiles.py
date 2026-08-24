"""Perfiles de wallets POL obtenidos desde swaps Uniswap V3 en Polygon.

El módulo parte del flujo Alchemy compartido: descarga eventos ``Swap`` de un
pool, los normaliza, construye ciclos FIFO y calcula etiquetas forward.  Las
funciones analíticas no necesitan red y aceptan ``DataFrame`` sintéticos, lo
que permite probarlas de forma determinista.

``SELL_POL`` mide el beneficio de vender frente a conservar POL; no representa
una posición corta. ``HOLD`` sólo se observa en lotes compra->venta completos,
por lo que balances anteriores al primer swap y transferencias externas no se
interpretan como posiciones conocidas.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

from pol_wallet_winners import (
    calcular_consistency_score,
    clasificar_winner_status,
    filtrar_wallets_ganadoras,
)


TICKER_POL = "POL-USD"
FALLBACK_PRICE_TICKERS = ("MATIC-USD",)
DEFAULT_POOL_ADDRESS = "0xA374094527e1673A86dE625aa59517c5dE346d32"
DEFAULT_LOOKBACK_HOURS = 24.0
DEFAULT_BLOCK_SPAN = 10  # Límite conservador de eth_getLogs en Alchemy Free.
DEFAULT_HORIZONS = ("1h", "4h", "1d")
HORIZONS: Mapping[str, pd.Timedelta] = {
    "1h": pd.Timedelta(hours=1),
    "4h": pd.Timedelta(hours=4),
    "1d": pd.Timedelta(days=1),
}
POOL_ABI_SWAP = [{
    "anonymous": False,
    "inputs": [
        {"indexed": True, "internalType": "address", "name": "sender", "type": "address"},
        {"indexed": True, "internalType": "address", "name": "recipient", "type": "address"},
        {"indexed": False, "internalType": "int256", "name": "amount0", "type": "int256"},
        {"indexed": False, "internalType": "int256", "name": "amount1", "type": "int256"},
        {"indexed": False, "internalType": "uint160", "name": "sqrtPriceX96", "type": "uint160"},
        {"indexed": False, "internalType": "uint128", "name": "liquidity", "type": "uint128"},
        {"indexed": False, "internalType": "int24", "name": "tick", "type": "int24"},
    ],
    "name": "Swap",
    "type": "event",
}]


def _require_web3() -> tuple[Any, Any]:
    """Carga web3 sólo al usar la red, para conservar importaciones testeables."""
    try:
        from web3 import Web3
        from web3.middleware import ExtraDataToPOAMiddleware
    except ImportError as exc:  # pragma: no cover - depende del entorno local.
        raise RuntimeError("Instala las dependencias: pip install -r requirements.txt") from exc
    return Web3, ExtraDataToPOAMiddleware


def conectar_polygon(rpc_url: str) -> Any:
    """Crea un cliente Polygon de Alchemy sin persistir ni imprimir la clave."""
    web3, poa_middleware = _require_web3()
    w3 = web3(web3.HTTPProvider(rpc_url))
    w3.middleware_onion.inject(poa_middleware, layer=0)
    if not w3.is_connected():
        raise ConnectionError("No fue posible conectar con el RPC de Polygon.")
    return w3


def obtener_contrato_pool(w3: Any, pool_address: str) -> Any:
    return w3.eth.contract(address=w3.to_checksum_address(pool_address), abi=POOL_ABI_SWAP)


def _as_utc(value: datetime | pd.Timestamp | str | None) -> pd.Timestamp:
    stamp = pd.Timestamp.now(tz="UTC") if value is None else pd.Timestamp(value)
    if stamp.tzinfo is None:
        stamp = stamp.tz_localize("UTC")
    else:
        stamp = stamp.tz_convert("UTC")
    return stamp


def _bloque_por_timestamp(w3: Any, timestamp_objetivo: int, bloque_max: int) -> int:
    """Devuelve el último bloque con timestamp <= objetivo mediante búsqueda binaria."""
    lo, hi = 0, bloque_max
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if int(w3.eth.get_block(mid)["timestamp"]) <= timestamp_objetivo:
            lo = mid
        else:
            hi = mid - 1
    return lo


def descargar_swaps(
    w3: Any,
    pool_contract: Any,
    lookback_hours: float = DEFAULT_LOOKBACK_HOURS,
    *,
    as_of: datetime | pd.Timestamp | str | None = None,
    block_span: int = DEFAULT_BLOCK_SPAN,
    pausa_seg: float = 0.05,
    verbose: bool = True,
) -> list[Any]:
    """Descarga swaps de una ventana acotada respetando rangos de 10 bloques.

    ``as_of`` hace reproducible el corte temporal. Para un corte histórico se
    resuelve primero el bloque final por timestamp; para el corte actual se
    usa directamente la cabeza de la cadena.
    """
    if lookback_hours <= 0:
        raise ValueError("lookback_hours debe ser positivo.")
    if not 1 <= block_span <= DEFAULT_BLOCK_SPAN:
        raise ValueError(f"block_span debe estar entre 1 y {DEFAULT_BLOCK_SPAN}.")

    cutoff = _as_utc(as_of)
    start = cutoff - pd.Timedelta(hours=lookback_hours)
    latest_block = int(w3.eth.block_number)
    latest_timestamp = int(w3.eth.get_block(latest_block)["timestamp"])
    block_end = (
        latest_block
        if abs(latest_timestamp - int(cutoff.timestamp())) < 120
        else _bloque_por_timestamp(w3, int(cutoff.timestamp()), latest_block)
    )
    block_start = _bloque_por_timestamp(w3, int(start.timestamp()), block_end)
    total_blocks = block_end - block_start + 1

    if verbose:
        print(
            f"[alchemy] {start.isoformat()} -> {cutoff.isoformat()} | "
            f"bloques {block_start}-{block_end} ({total_blocks:,})"
        )

    events: list[Any] = []
    total_batches = math.ceil(total_blocks / block_span)
    for batch_index, first in enumerate(range(block_start, block_end + 1, block_span), start=1):
        last = min(first + block_span - 1, block_end)
        try:
            events.extend(pool_contract.events.Swap.get_logs(from_block=first, to_block=last))
        except Exception as exc:  # La ejecución conserva lotes parciales y registra el alcance.
            if verbose:
                print(f"[alchemy] error en bloques {first}-{last}: {exc}")
        if verbose and (batch_index == total_batches or batch_index % 100 == 0):
            print(f"  lote {batch_index}/{total_batches}; swaps: {len(events):,}")
        if pausa_seg:
            time.sleep(pausa_seg)
    return events


def _event_log_index(event: Any) -> int:
    value = event.get("logIndex", event.get("log_index", -1))
    return int(value)


def _enriquecer_evento(
    w3: Any,
    swap: Any,
    *,
    decimales_t0: int = 18,
    decimales_t1: int = 6,
    block_cache: dict[int, Any] | None = None,
    tx_cache: dict[str, Any] | None = None,
    receipt_cache: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Convierte un evento Swap en una fila sin repetir lecturas RPC por hash."""
    args = swap["args"]
    block_number = int(swap["blockNumber"])
    tx_hash = swap["transactionHash"].hex()
    if not tx_hash.startswith("0x"):
        tx_hash = "0x" + tx_hash
    block_cache = block_cache if block_cache is not None else {}
    tx_cache = tx_cache if tx_cache is not None else {}
    receipt_cache = receipt_cache if receipt_cache is not None else {}

    if block_number not in block_cache:
        block_cache[block_number] = w3.eth.get_block(block_number)
    if tx_hash not in tx_cache:
        tx_cache[tx_hash] = w3.eth.get_transaction(tx_hash)
    if tx_hash not in receipt_cache:
        receipt_cache[tx_hash] = w3.eth.get_transaction_receipt(tx_hash)
    block = block_cache[block_number]
    transaction = tx_cache[tx_hash]
    receipt = receipt_cache[tx_hash]
    timestamp = pd.Timestamp(int(block["timestamp"]), unit="s", tz="UTC")
    effective_gas_price = receipt.get("effectiveGasPrice", transaction.get("gasPrice", 0))
    gas_pol = float(receipt["gasUsed"] * effective_gas_price) / 1e18

    pol_signed = float(args["amount0"]) / 10**decimales_t0
    usdc_signed = float(args["amount1"]) / 10**decimales_t1
    pol_amount = abs(pol_signed)
    usdc_amount = abs(usdc_signed)
    execution_price = usdc_amount / pol_amount if pol_amount else np.nan
    action = "BUY_POL" if pol_signed < 0 else "SELL_POL" if pol_signed > 0 else "AMBIGUOUS"
    return {
        "timestamp": timestamp,
        "block_number": block_number,
        "log_index": _event_log_index(swap),
        "hash_tx": tx_hash.lower(),
        "wallet": str(transaction["from"]).lower(),
        "direccion": action,
        "pol_delta": -pol_signed,  # positivo cuando la wallet recibe POL.
        "usdc_delta": -usdc_signed,
        "pol_cantidad": pol_amount,
        "usdc_cantidad": usdc_amount,
        "precio_ejecutado": execution_price,
        "gas_pol": gas_pol,
        "gas_usdc": gas_pol * execution_price if pd.notna(execution_price) else np.nan,
    }


def construir_tabla_swaps(w3: Any, events: Iterable[Any], *, verbose: bool = True) -> pd.DataFrame:
    """Construye la tabla transaccional de eventos Swap enriquecidos."""
    rows: list[dict[str, Any]] = []
    block_cache: dict[int, Any] = {}
    tx_cache: dict[str, Any] = {}
    receipt_cache: dict[str, Any] = {}
    events_list = list(events)
    for index, event in enumerate(events_list, start=1):
        try:
            rows.append(_enriquecer_evento(
                w3, event, block_cache=block_cache, tx_cache=tx_cache, receipt_cache=receipt_cache
            ))
        except Exception as exc:
            if verbose:
                print(f"[alchemy] no se pudo enriquecer el swap {index}: {exc}")
        if verbose and (index == len(events_list) or index % 100 == 0):
            print(f"  enriquecidos {index}/{len(events_list)}")
    return pd.DataFrame(rows).sort_values(["timestamp", "hash_tx", "log_index"]).reset_index(drop=True) if rows else pd.DataFrame()


def _empty_logical_swaps() -> pd.DataFrame:
    return pd.DataFrame(columns=[
        "timestamp", "block_number", "hash_tx", "wallet", "direccion", "pol_delta",
        "usdc_delta", "pol_cantidad", "notional_usdc", "precio_ejecutado", "gas_usdc",
        "event_count",
    ])


def canonicalizar_swaps_logicos(swaps: pd.DataFrame) -> pd.DataFrame:
    """Agrupa eventos de una transacción y evita contar gas varias veces.

    El resultado expresa el cambio desde el punto de vista de la wallet:
    ``pol_delta > 0`` es una compra. Si una transacción deja POL neto cero se
    conserva con ``AMBIGUOUS`` para auditoría, pero no alimenta perfiles.
    """
    if swaps.empty:
        return _empty_logical_swaps()
    frame = swaps.copy()
    # Compatibilidad con ``swaps_10h.parquet`` producido por el pipeline
    # original compartido: usaba Buy/Sell, bloque y cantidades absolutas.
    if "block_number" not in frame.columns:
        frame["block_number"] = frame["bloque"] if "bloque" in frame.columns else pd.NA
    if "log_index" not in frame.columns:
        frame["log_index"] = 0
    if "pol_delta" not in frame.columns and {"direccion", "pol_cantidad"}.issubset(frame.columns):
        direction = frame["direccion"].astype(str).str.upper()
        frame["pol_delta"] = np.select(
            [direction.isin(["BUY", "BUY_POL"]), direction.isin(["SELL", "SELL_POL"])],
            [frame["pol_cantidad"], -frame["pol_cantidad"]],
            default=0.0,
        )
    if "usdc_delta" not in frame.columns and {"direccion", "usdc_cantidad"}.issubset(frame.columns):
        direction = frame["direccion"].astype(str).str.upper()
        frame["usdc_delta"] = np.select(
            [direction.isin(["BUY", "BUY_POL"]), direction.isin(["SELL", "SELL_POL"])],
            [-frame["usdc_cantidad"], frame["usdc_cantidad"]],
            default=0.0,
        )
    if "gas_usdc" not in frame.columns:
        frame["gas_usdc"] = 0.0
    required = {"timestamp", "hash_tx", "wallet", "pol_delta", "usdc_delta", "gas_usdc"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Faltan columnas en swaps: {sorted(missing)}")

    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    frame["hash_tx"] = frame["hash_tx"].astype(str).str.lower()
    frame["wallet"] = frame["wallet"].astype(str).str.lower()
    aggregations: dict[str, Any] = {
        "timestamp": "min",
        "block_number": "min",
        "pol_delta": "sum",
        "usdc_delta": "sum",
        "gas_usdc": "max",
        "log_index": "count",
    }
    if "precio_ejecutado" in frame.columns:
        aggregations["precio_ejecutado"] = "median"
    grouped = frame.groupby(["wallet", "hash_tx"], as_index=False).agg(aggregations)
    grouped = grouped.rename(columns={"log_index": "event_count"})
    grouped["pol_cantidad"] = grouped["pol_delta"].abs()
    grouped["notional_usdc"] = grouped["usdc_delta"].abs()
    grouped["precio_ejecutado"] = np.where(
        grouped["pol_cantidad"] > 0,
        grouped["notional_usdc"] / grouped["pol_cantidad"],
        grouped.get("precio_ejecutado", np.nan),
    )
    grouped["direccion"] = np.select(
        [grouped["pol_delta"] > 0, grouped["pol_delta"] < 0],
        ["BUY_POL", "SELL_POL"],
        default="AMBIGUOUS",
    )
    grouped["gas_usdc"] = grouped["gas_usdc"].fillna(0.0)
    return grouped.sort_values(["timestamp", "hash_tx"]).reset_index(drop=True)[_empty_logical_swaps().columns]


def construir_ciclos_hold(swaps_logicos: pd.DataFrame) -> pd.DataFrame:
    """Reconstruye lotes FIFO de compra a venta y marca inventario no observable."""
    columns = [
        "cycle_id", "wallet", "buy_hash_tx", "sell_hash_tx", "opened_at", "closed_at",
        "pol_cantidad", "entry_price", "exit_price", "entry_gas_usdc", "exit_gas_usdc",
        "duration_hours", "pnl_realizado_usdc", "cycle_status",
    ]
    if swaps_logicos.empty:
        return pd.DataFrame(columns=columns)
    trades = swaps_logicos.copy()
    trades["timestamp"] = pd.to_datetime(trades["timestamp"], utc=True)
    trades = trades[trades["direccion"].isin(["BUY_POL", "SELL_POL"])].sort_values(
        ["wallet", "timestamp", "hash_tx"]
    )
    cycles: list[dict[str, Any]] = []
    sequence = 0
    for wallet, wallet_trades in trades.groupby("wallet", sort=False):
        open_lots: deque[dict[str, Any]] = deque()
        for _, trade in wallet_trades.iterrows():
            quantity = float(trade["pol_cantidad"])
            if quantity <= 0:
                continue
            if trade["direccion"] == "BUY_POL":
                open_lots.append({
                    "remaining": quantity,
                    "original": quantity,
                    "opened_at": trade["timestamp"],
                    "hash_tx": trade["hash_tx"],
                    "price": float(trade["precio_ejecutado"]),
                    "gas_per_pol": float(trade.get("gas_usdc", 0.0)) / quantity,
                })
                continue

            remaining = quantity
            sell_gas_per_pol = float(trade.get("gas_usdc", 0.0)) / quantity
            while remaining > 1e-12 and open_lots:
                lot = open_lots[0]
                matched = min(remaining, lot["remaining"])
                sequence += 1
                entry_gas = matched * lot["gas_per_pol"]
                exit_gas = matched * sell_gas_per_pol
                cycles.append({
                    "cycle_id": f"{wallet}:{sequence}",
                    "wallet": wallet,
                    "buy_hash_tx": lot["hash_tx"],
                    "sell_hash_tx": trade["hash_tx"],
                    "opened_at": lot["opened_at"],
                    "closed_at": trade["timestamp"],
                    "pol_cantidad": matched,
                    "entry_price": lot["price"],
                    "exit_price": float(trade["precio_ejecutado"]),
                    "entry_gas_usdc": entry_gas,
                    "exit_gas_usdc": exit_gas,
                    "duration_hours": (trade["timestamp"] - lot["opened_at"]).total_seconds() / 3600,
                    "pnl_realizado_usdc": matched * (float(trade["precio_ejecutado"]) - lot["price"]) - entry_gas - exit_gas,
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
                    "opened_at": pd.NaT, "closed_at": trade["timestamp"],
                    "pol_cantidad": remaining, "entry_price": np.nan,
                    "exit_price": float(trade["precio_ejecutado"]), "entry_gas_usdc": np.nan,
                    "exit_gas_usdc": remaining * sell_gas_per_pol, "duration_hours": np.nan,
                    "pnl_realizado_usdc": np.nan, "cycle_status": "CENSORED_PREEXISTING",
                })
        for lot in open_lots:
            sequence += 1
            cycles.append({
                "cycle_id": f"{wallet}:{sequence}", "wallet": wallet,
                "buy_hash_tx": lot["hash_tx"], "sell_hash_tx": None,
                "opened_at": lot["opened_at"], "closed_at": pd.NaT,
                "pol_cantidad": lot["remaining"], "entry_price": lot["price"],
                "exit_price": np.nan, "entry_gas_usdc": lot["remaining"] * lot["gas_per_pol"],
                "exit_gas_usdc": np.nan, "duration_hours": np.nan,
                "pnl_realizado_usdc": np.nan, "cycle_status": "CENSORED_OPEN",
            })
    return pd.DataFrame(cycles, columns=columns).sort_values(["wallet", "opened_at", "closed_at"], na_position="last").reset_index(drop=True)


def _normalizar_precios(precios: pd.DataFrame) -> pd.Series:
    """Normaliza Close de yfinance o una tabla timestamp/price al horario UTC."""
    if precios.empty:
        return pd.Series(dtype=float, name="price")
    frame = precios.copy()
    if "timestamp" in frame.columns:
        index = pd.to_datetime(frame.pop("timestamp"), utc=True)
        values = frame["price"] if "price" in frame.columns else frame["Close"]
    else:
        index = pd.to_datetime(frame.index, utc=True)
        values = frame["price"] if "price" in frame.columns else frame["Close"]
    series = pd.Series(pd.to_numeric(values, errors="coerce").to_numpy(), index=index, name="price").dropna()
    return series.groupby(series.index.floor("h")).last().sort_index()


def descargar_precios_horarios(
    *,
    as_of: datetime | pd.Timestamp | str | None = None,
    lookback_hours: float = DEFAULT_LOOKBACK_HOURS,
    ticker: str = TICKER_POL,
    fallback_tickers: Iterable[str] = FALLBACK_PRICE_TICKERS,
) -> pd.DataFrame:
    """Obtiene velas horarias conocidas al corte; nunca solicita precios futuros.

    Yahoo no siempre publica el ticker ``POL-USD`` ni velas intradía para él.
    Si no hay datos se intenta ``MATIC-USD`` como proxy histórico 1:1 de POL.
    Un fallo de Yahoo no detiene la reconstrucción on-chain: las etiquetas se
    conservan como ``MISSING_PRICE`` hasta disponer de una fuente de precio.
    """
    try:
        import yfinance as yf
    except ImportError as exc:  # pragma: no cover - depende del entorno local.
        raise RuntimeError("Instala yfinance para descargar precios forward.") from exc
    cutoff = _as_utc(as_of)
    start = cutoff - pd.Timedelta(hours=lookback_hours + max(HORIZONS.values()).total_seconds() / 3600 + 2)
    candidates = tuple(dict.fromkeys((ticker, *fallback_tickers)))
    for candidate in candidates:
        try:
            candles = yf.download(
                candidate, start=start.to_pydatetime(), end=cutoff.to_pydatetime(), interval="1h",
                progress=False, auto_adjust=True, threads=False,
            )
        except Exception as exc:  # yfinance puede fallar por ticker, rate limit o red.
            print(f"[precio] Yahoo no respondió para {candidate}: {exc}")
            continue
        if candles is None or candles.empty:
            print(f"[precio] sin velas 1h para {candidate}; probando el siguiente ticker.")
            continue
        if isinstance(candles.columns, pd.MultiIndex):
            candles.columns = candles.columns.get_level_values(0)
        candles.index = pd.to_datetime(candles.index, utc=True)
        result = candles[["Close"]].dropna().copy()
        if not result.empty:
            result["source_ticker"] = candidate
            if candidate != ticker:
                print(f"[precio] usando {candidate} como proxy histórico de {ticker}.")
            return result
    print("[precio] no hubo velas disponibles; las etiquetas forward quedarán MISSING_PRICE.")
    return pd.DataFrame(columns=["Close", "source_ticker"])


def _label_row(
    *,
    action: str,
    decision_price: float,
    notional_usdc: float,
    gas_usdc: float,
    decision_time: pd.Timestamp,
    horizon: str,
    prices: pd.Series,
    cutoff: pd.Timestamp,
) -> dict[str, Any]:
    delta = HORIZONS[horizon]
    target = decision_time.floor("h") + delta
    base = {
        "horizonte": horizon,
        "forward_time": target,
        "forward_price": np.nan,
        "retorno_bruto": np.nan,
        "retorno_neto": np.nan,
        "ganancia_neta_usdc": np.nan,
    }
    if target > cutoff.floor("h"):
        return {**base, "maturity_status": "PENDING"}
    if target not in prices.index or not pd.notna(decision_price) or decision_price <= 0:
        return {**base, "maturity_status": "MISSING_PRICE"}
    forward_price = float(prices.loc[target])
    if action == "BUY_POL":
        gross = (forward_price - decision_price) / decision_price
    elif action == "SELL_POL":
        gross = (decision_price - forward_price) / decision_price
    elif action == "HOLD":
        gross = (forward_price - decision_price) / decision_price
    else:
        raise ValueError(f"Acción no soportada: {action}")
    fee_return = 0.0 if action == "HOLD" else gas_usdc / notional_usdc if notional_usdc > 0 else np.nan
    net = gross - fee_return
    return {
        **base,
        "forward_price": forward_price,
        "retorno_bruto": gross,
        "retorno_neto": net,
        "ganancia_neta_usdc": net * notional_usdc,
        "maturity_status": "COMPLETED",
    }


def construir_ledger_decisiones(
    swaps_logicos: pd.DataFrame,
    ciclos_hold: pd.DataFrame,
    precios: pd.DataFrame,
    *,
    as_of: datetime | pd.Timestamp | str | None = None,
    horizontes: Iterable[str] = DEFAULT_HORIZONS,
) -> pd.DataFrame:
    """Crea observaciones Buy/Sell/Hold en formato largo y sin fuga futura."""
    requested = tuple(horizontes)
    invalid = set(requested).difference(HORIZONS)
    if invalid:
        raise ValueError(f"Horizontes inválidos: {sorted(invalid)}")
    cutoff = _as_utc(as_of)
    prices = _normalizar_precios(precios)
    records: list[dict[str, Any]] = []
    trades = swaps_logicos[swaps_logicos["direccion"].isin(["BUY_POL", "SELL_POL"])].copy()
    for _, trade in trades.iterrows():
        decision_time = _as_utc(trade["timestamp"])
        for horizon in requested:
            record = {
                "decision_id": f"trade:{trade['wallet']}:{trade['hash_tx']}:{horizon}",
                "wallet": trade["wallet"], "accion": trade["direccion"], "horizonte": horizon,
                "source_type": "TRADE", "source_id": trade["hash_tx"],
                "decision_time": decision_time, "decision_price": float(trade["precio_ejecutado"]),
                "notional_usdc": float(trade["notional_usdc"]), "gas_usdc": float(trade.get("gas_usdc", 0.0)),
            }
            records.append({**record, **_label_row(
                action=record["accion"], decision_price=record["decision_price"],
                notional_usdc=record["notional_usdc"], gas_usdc=record["gas_usdc"],
                decision_time=decision_time, horizon=horizon, prices=prices, cutoff=cutoff,
            )})
    if not ciclos_hold.empty:
        closed = ciclos_hold[ciclos_hold["cycle_status"] == "CLOSED"]
        for _, cycle in closed.iterrows():
            decision_time = _as_utc(cycle["opened_at"])
            for horizon in requested:
                if float(cycle["duration_hours"]) + 1e-12 < HORIZONS[horizon].total_seconds() / 3600:
                    continue
                notional = float(cycle["pol_cantidad"] * cycle["entry_price"])
                record = {
                    "decision_id": f"hold:{cycle['cycle_id']}:{horizon}",
                    "wallet": cycle["wallet"], "accion": "HOLD", "horizonte": horizon,
                    "source_type": "HOLD_CYCLE", "source_id": cycle["cycle_id"],
                    "decision_time": decision_time, "decision_price": float(cycle["entry_price"]),
                    "notional_usdc": notional, "gas_usdc": 0.0,
                }
                records.append({**record, **_label_row(
                    action="HOLD", decision_price=record["decision_price"], notional_usdc=notional,
                    gas_usdc=0.0, decision_time=decision_time, horizon=horizon,
                    prices=prices, cutoff=cutoff,
                )})
    columns = [
        "decision_id", "wallet", "accion", "horizonte", "source_type", "source_id",
        "decision_time", "decision_price", "notional_usdc", "gas_usdc", "forward_time",
        "forward_price", "retorno_bruto", "retorno_neto", "ganancia_neta_usdc", "maturity_status",
    ]
    return pd.DataFrame(records, columns=columns).sort_values(["decision_time", "wallet", "decision_id"]).reset_index(drop=True) if records else pd.DataFrame(columns=columns)


def _profit_factor(values: pd.Series) -> float:
    positive = float(values.clip(lower=0).sum())
    negative = float((-values.clip(upper=0)).sum())
    if negative == 0:
        return 3.0 if positive > 0 else 0.0
    return min(positive / negative, 3.0)


def construir_perfiles_wallet(
    ledger: pd.DataFrame,
    swaps_logicos: pd.DataFrame,
    ciclos_hold: pd.DataFrame,
    *,
    min_decisiones: int = 3,
) -> pd.DataFrame:
    """Resume rentabilidad, evidencia y rasgos interpretable por wallet/acción/horizonte."""
    if min_decisiones < 1:
        raise ValueError("min_decisiones debe ser >= 1.")
    columns = [
        "wallet", "accion", "horizonte", "n_decisiones", "n_pendientes", "pnl_neto_usdc",
        "retorno_neto_mediano", "tasa_acierto", "profit_factor", "consistency_score",
        "winner_status", "perfil_conductual", "duration_hold_mediana_horas",
    ]
    if ledger.empty:
        return pd.DataFrame(columns=columns)
    closed_cycles = ciclos_hold[ciclos_hold["cycle_status"] == "CLOSED"] if not ciclos_hold.empty else ciclos_hold
    durations = (
        closed_cycles.groupby("wallet")["duration_hours"].median().rename("duration_hold_mediana_horas")
        if not closed_cycles.empty else pd.Series(dtype=float, name="duration_hold_mediana_horas")
    )
    active = swaps_logicos[swaps_logicos["direccion"].isin(["BUY_POL", "SELL_POL"])].copy()
    flow = active.groupby("wallet")["pol_delta"].sum() if not active.empty else pd.Series(dtype=float)

    records: list[dict[str, Any]] = []
    for (wallet, action, horizon), group in ledger.groupby(["wallet", "accion", "horizonte"], dropna=False):
        completed = group[group["maturity_status"] == "COMPLETED"]
        n = len(completed)
        pending = int((group["maturity_status"] == "PENDING").sum())
        if n:
            pnl = float(completed["ganancia_neta_usdc"].sum())
            median_return = float(completed["retorno_neto"].median())
            hit_rate = float((completed["retorno_neto"] > 0).mean())
            profit_factor = _profit_factor(completed["ganancia_neta_usdc"])
            score = calcular_consistency_score(hit_rate, profit_factor, n)
        else:
            pnl = median_return = hit_rate = profit_factor = score = np.nan
        status = clasificar_winner_status(
            n_decisiones=n,
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
            "wallet": wallet, "accion": action, "horizonte": horizon, "n_decisiones": n,
            "n_pendientes": pending, "pnl_neto_usdc": pnl, "retorno_neto_mediano": median_return,
            "tasa_acierto": hit_rate, "profit_factor": profit_factor, "consistency_score": score,
            "winner_status": status, "perfil_conductual": f"{bias}_{tempo}",
            "duration_hold_mediana_horas": duration,
        })
    return pd.DataFrame(records, columns=columns).sort_values(["winner_status", "consistency_score"], ascending=[True, False], na_position="last").reset_index(drop=True)


def construir_estado_rl(
    swaps_logicos: pd.DataFrame,
    ciclos_hold: pd.DataFrame,
    perfiles: pd.DataFrame,
    *,
    as_of: datetime | pd.Timestamp | str | None = None,
    ventanas: Iterable[str] = DEFAULT_HORIZONS,
) -> pd.DataFrame:
    """Genera variables observables del agente, limitadas a información <= as_of."""
    cutoff = _as_utc(as_of)
    windows = tuple(ventanas)
    if set(windows).difference(HORIZONS):
        raise ValueError("ventanas debe contener sólo 1h, 4h y/o 1d.")
    active = swaps_logicos[swaps_logicos["timestamp"] <= cutoff].copy() if not swaps_logicos.empty else swaps_logicos
    wallets = sorted(set(active.get("wallet", pd.Series(dtype=str)).dropna()) | set(perfiles.get("wallet", pd.Series(dtype=str)).dropna()))
    records: list[dict[str, Any]] = []
    closed = ciclos_hold[ciclos_hold["cycle_status"] == "CLOSED"] if not ciclos_hold.empty else ciclos_hold
    for wallet in wallets:
        wallet_trades = active[active["wallet"] == wallet]
        for window in windows:
            start = cutoff - HORIZONS[window]
            recent = wallet_trades[wallet_trades["timestamp"] > start]
            buy_count = int((recent["direccion"] == "BUY_POL").sum())
            sell_count = int((recent["direccion"] == "SELL_POL").sum())
            wallet_cycles = closed[(closed["wallet"] == wallet) & (closed["closed_at"] <= cutoff)] if not closed.empty else closed
            record: dict[str, Any] = {
                "as_of": cutoff, "wallet": wallet, "ventana": window,
                "n_swaps": int(len(recent)), "buy_count": buy_count, "sell_count": sell_count,
                "ratio_buy_sell": buy_count / max(sell_count, 1),
                "notional_usdc": float(recent.get("notional_usdc", pd.Series(dtype=float)).sum()),
                "flujo_neto_pol": float(recent.get("pol_delta", pd.Series(dtype=float)).sum()),
                "n_ciclos_cerrados": int(len(wallet_cycles)),
                "duration_hold_media_horas": float(wallet_cycles["duration_hours"].mean()) if len(wallet_cycles) else np.nan,
                "duration_hold_mediana_horas": float(wallet_cycles["duration_hours"].median()) if len(wallet_cycles) else np.nan,
            }
            for action in ("BUY_POL", "SELL_POL", "HOLD"):
                subset = perfiles[(perfiles["wallet"] == wallet) & (perfiles["accion"] == action) & (perfiles["horizonte"] == window)]
                prefix = action.lower()
                if subset.empty:
                    record[f"score_{prefix}"] = np.nan
                    record[f"winner_{prefix}"] = False
                else:
                    profile = subset.iloc[0]
                    record[f"score_{prefix}"] = profile["consistency_score"]
                    record[f"winner_{prefix}"] = profile["winner_status"] == "winner"
            records.append(record)
    return pd.DataFrame(records)


def crear_figuras_perfiles(
    perfiles: pd.DataFrame,
    ledger: pd.DataFrame,
    ciclos_hold: pd.DataFrame,
    swaps_logicos: pd.DataFrame,
) -> dict[str, Any]:
    """Devuelve cinco gráficas Plotly; no escribe archivos ni requiere datos de red."""
    try:
        import plotly.express as px
        import plotly.graph_objects as go
    except ImportError as exc:  # pragma: no cover - depende del entorno local.
        raise RuntimeError("Instala plotly y kaleido para crear las gráficas.") from exc

    def empty(title: str) -> Any:
        figure = go.Figure()
        figure.add_annotation(text="Sin datos para esta vista", showarrow=False, font={"size": 18})
        figure.update_layout(title=title, template="plotly_white")
        return figure

    def short_wallet(value: Any) -> str:
        text = str(value)
        return f"{text[:6]}…{text[-4:]}" if len(text) > 14 else text

    def top_bottom(frame: pd.DataFrame, metric: str, *, n: int = 5) -> pd.DataFrame:
        """Selecciona extremos sin mezclar orden descendente y ascendente."""
        source = frame.dropna(subset=[metric]).copy()
        if source.empty:
            return source.assign(grupo_extremo=pd.Series(dtype=str))
        top = source.nlargest(n, metric).sort_values(metric, ascending=False).assign(
            grupo_extremo="Top 5 · descendente"
        )
        bottom = source.nsmallest(n, metric).sort_values(metric, ascending=True).assign(
            grupo_extremo="Bottom 5 · ascendente"
        )
        # Con menos de 10 filas no se duplica una misma wallet en ambos paneles.
        return pd.concat([top, bottom]).drop_duplicates(
            [column for column in ("wallet", "accion", "horizonte") if column in source.columns],
            keep="first",
        )

    activity = pd.DataFrame(columns=["wallet", "direccion", "n_swaps", "notional_usdc"])
    if not swaps_logicos.empty:
        activity = swaps_logicos[swaps_logicos["direccion"].isin(["BUY_POL", "SELL_POL"])].groupby(
            ["wallet", "direccion"], as_index=False
        ).agg(n_swaps=("hash_tx", "count"), notional_usdc=("notional_usdc", "sum"))

    if perfiles.empty:
        ranking = scatter = heatmap = empty("Sin perfiles disponibles")
    else:
        scored = filtrar_wallets_ganadoras(perfiles).dropna(subset=["consistency_score"]).copy()
        if scored.empty:
            status_counts = perfiles.groupby(["accion", "horizonte", "winner_status"], as_index=False).size()
            status_counts = status_counts.rename(columns={"size": "n_wallets"})
            activity_total = activity.groupby("wallet", as_index=False).agg(n_swaps=("n_swaps", "sum"))
            selected = top_bottom(activity_total, "n_swaps")
            ranking_data = activity.merge(selected[["wallet", "grupo_extremo"]], on="wallet", how="inner")
            ranking_data["wallet_corta"] = ranking_data["wallet"].map(short_wallet)
            if ranking_data.empty:
                ranking = empty("Actividad pendiente: no hubo swaps lógicos")
            else:
                ranking = px.bar(
                    ranking_data, x="n_swaps", y="wallet_corta", color="direccion", orientation="h",
                    facet_col="grupo_extremo", barmode="stack", hover_data=["wallet", "notional_usdc"],
                    title="Top 5 y Bottom 5 por actividad de swaps",
                    labels={"n_swaps": "Número de swaps", "wallet_corta": "Wallet"}, template="plotly_white",
                )
                ranking.for_each_yaxis(lambda axis: axis.update(autorange="reversed"))
                pending_count = int((status_counts["winner_status"] == "pending").sum())
                ranking.add_annotation(
                    text=f"{pending_count} grupos aún esperan precio forward; este ranking usa actividad, no rentabilidad.",
                    x=0.5, y=1.14, xref="paper", yref="paper", showarrow=False, font={"color": "#475569"},
                )
            scatter = empty("Mapa retorno–consistencia: pendiente de precios forward")
            matrix = perfiles.assign(column=perfiles["accion"] + " / " + perfiles["horizonte"])
            pivot = matrix.pivot_table(index="wallet", columns="column", values="winner_status", aggfunc="first")
            status_value = {"pending": 0, "insufficient": 1, "not_winner": 2, "winner": 3}
            status_text = pivot.fillna("sin datos")
            heatmap = go.Figure(go.Heatmap(
                z=status_text.replace(status_value).fillna(-1).to_numpy(),
                text=status_text.to_numpy(), texttemplate="%{text}",
                x=pivot.columns.tolist(), y=[short_wallet(wallet) for wallet in pivot.index],
                colorscale=[(0, "#cbd5e1"), (0.25, "#f59e0b"), (0.5, "#64748b"), (0.75, "#ef4444"), (1, "#16a34a")],
                zmin=-1, zmax=3, colorbar={"tickvals": [-1, 0, 1, 2, 3], "ticktext": ["sin datos", "pending", "insufficient", "not winner", "winner"]},
            ))
            heatmap.update_layout(title="Estado por wallet × acción × horizonte", template="plotly_white")
        else:
            ranking_data = top_bottom(scored, "consistency_score")
            ranking_data["wallet_corta"] = (
                ranking_data["wallet"].map(short_wallet)
                + " · " + ranking_data["accion"].str.replace("_POL", "", regex=False)
                + " · " + ranking_data["horizonte"]
            )
            ranking = px.bar(
                ranking_data, x="consistency_score", y="wallet_corta", color="winner_status", orientation="h",
                facet_col="grupo_extremo", hover_data=["wallet", "pnl_neto_usdc", "n_decisiones", "tasa_acierto"],
                title="Top 5 y Bottom 5 por consistencia", template="plotly_white",
            )
            ranking.for_each_yaxis(lambda axis: axis.update(autorange="reversed"))
            scatter = px.scatter(
                scored, x="retorno_neto_mediano", y="tasa_acierto", size="n_decisiones",
                color="winner_status", facet_col="horizonte", facet_row="accion",
                hover_data=["wallet", "perfil_conductual", "pnl_neto_usdc", "consistency_score"],
                title="Mapa retorno–consistencia", template="plotly_white",
            )
            matrix = scored.assign(column=scored["accion"] + " / " + scored["horizonte"])
            pivot = matrix.pivot_table(index="wallet", columns="column", values="consistency_score", aggfunc="first")
            heatmap = go.Figure(go.Heatmap(
                z=pivot.to_numpy(), x=pivot.columns.tolist(), y=[short_wallet(wallet) for wallet in pivot.index], colorscale="Viridis",
                colorbar_title="Score",
            ))
            heatmap.update_layout(title="Matriz wallet × acción × horizonte", template="plotly_white")
    closed = ciclos_hold[ciclos_hold["cycle_status"] == "CLOSED"] if not ciclos_hold.empty else ciclos_hold
    if closed.empty:
        timeline = empty("Ciclos Hold")
    else:
        top_wallets = closed.groupby("wallet")["pnl_realizado_usdc"].sum().nlargest(15).index
        timeline_data = closed[closed["wallet"].isin(top_wallets)].copy()
        timeline_data["wallet_corta"] = timeline_data["wallet"].map(short_wallet)
        timeline = px.timeline(
            timeline_data, x_start="opened_at", x_end="closed_at", y="wallet_corta",
            color="pnl_realizado_usdc", hover_data=["pol_cantidad", "duration_hours", "cycle_status"],
            title="Línea de tiempo de ciclos Hold", template="plotly_white",
        )
        timeline.update_yaxes(autorange="reversed")
    if activity.empty:
        behavior = empty("Composición conductual")
    else:
        volume_total = activity.groupby("wallet", as_index=False).agg(notional_usdc=("notional_usdc", "sum"))
        selected = top_bottom(volume_total, "notional_usdc")
        composition = activity.merge(selected[["wallet", "grupo_extremo"]], on="wallet", how="inner")
        composition["wallet_corta"] = composition["wallet"].map(short_wallet)
        behavior = px.bar(
            composition, x="notional_usdc", y="wallet_corta", color="direccion", barmode="stack",
            orientation="h", facet_col="grupo_extremo", hover_data=["wallet", "n_swaps"],
            title="Top 5 y Bottom 5 por volumen Buy/Sell", template="plotly_white",
        )
        behavior.for_each_yaxis(lambda axis: axis.update(autorange="reversed"))
    return {"ranking": ranking, "mapa_retorno": scatter, "matriz": heatmap, "ciclos_hold": timeline, "comportamiento": behavior}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)


def exportar_snapshot_derivado(
    *,
    swaps_logicos: pd.DataFrame,
    ciclos_hold: pd.DataFrame,
    ledger: pd.DataFrame,
    perfiles: pd.DataFrame,
    estado_rl: pd.DataFrame,
    source_paths: Iterable[str | Path] = (),
    as_of: datetime | pd.Timestamp | str | None = None,
    output_dir: str | Path = "data/derived/alchemy_wallet_profiles",
    lookback_hours: float = DEFAULT_LOOKBACK_HOURS,
    export_png: bool = True,
) -> Path:
    """Persiste artefactos, manifiesto y gráficas en un snapshot inmutable."""
    cutoff = _as_utc(as_of)
    root = Path(output_dir) / f"snapshot_{cutoff.strftime('%Y%m%dT%H%M%SZ')}"
    if root.exists():
        raise FileExistsError(f"Ya existe el snapshot derivado {root}.")
    root.mkdir(parents=True)
    datasets = {
        "swaps_logicos.parquet": swaps_logicos,
        "ciclos_hold.parquet": ciclos_hold,
        "ledger_decisiones.parquet": ledger,
        "perfiles_wallet.parquet": perfiles,
        "estado_rl.parquet": estado_rl,
    }
    for name, frame in datasets.items():
        _write_parquet(frame, root / name)
    figures = crear_figuras_perfiles(perfiles, ledger, ciclos_hold, swaps_logicos)
    html_sections: list[str] = []
    png_export_errors: list[dict[str, str]] = []
    for name, figure in figures.items():
        figure.write_html(root / f"{name}.html", include_plotlyjs="cdn")
        html_sections.append(figure.to_html(full_html=False, include_plotlyjs="cdn" if not html_sections else False))
        if export_png:
            try:
                figure.write_image(root / f"{name}.png", scale=2)
            except (ImportError, OSError, ValueError) as exc:
                # HTML sigue siendo útil en Colab aunque Kaleido no esté instalado.
                png_export_errors.append({"figure": name, "error": str(exc)})
                print(f"[gráfica] no se pudo exportar {name}.png: {exc}")
    (root / "informe_wallets.html").write_text(
        "<html><head><meta charset='utf-8'><title>Perfiles POL</title></head><body>"
        + "\n".join(html_sections) + "</body></html>", encoding="utf-8"
    )
    sources = []
    for item in source_paths:
        path = Path(item)
        sources.append({"path": str(path), "sha256": _sha256_file(path) if path.is_file() else None})
    manifest = {
        "schema_version": 1,
        "source": "Alchemy + yfinance",
        "as_of_utc": cutoff.isoformat(),
        "lookback_hours": lookback_hours,
        "horizons": list(DEFAULT_HORIZONS),
        "min_decisions": 3,
        "png_export_errors": png_export_errors,
        "source_artifacts": sources,
        "artifacts": [{"path": name, "sha256": _sha256_file(root / name), "rows": int(len(frame))} for name, frame in datasets.items()],
        "semantics": {
            "sell": "Vender POL frente a conservarlo; no short.",
            "hold": "Sólo lotes FIFO compra->venta cerrados.",
            "pending": "El precio forward aún no era conocido al corte.",
        },
    }
    (root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return root


def guardar_cache_swaps(frame: pd.DataFrame, cache_dir: str | Path, as_of: datetime | pd.Timestamp | str | None = None) -> Path:
    """Guarda el snapshot bruto de 24 h; no sobrescribe extracciones anteriores."""
    cutoff = _as_utc(as_of)
    path = Path(cache_dir) / f"swaps_{cutoff.strftime('%Y%m%dT%H%M%SZ')}.parquet"
    if path.exists():
        raise FileExistsError(f"Ya existe {path}.")
    _write_parquet(frame, path)
    return path


def guardar_cache_precios(frame: pd.DataFrame, cache_dir: str | Path, as_of: datetime | pd.Timestamp | str | None = None) -> Path:
    """Guarda precios horarios solapados para cerrar etiquetas en ejecuciones posteriores."""
    cutoff = _as_utc(as_of)
    path = Path(cache_dir) / f"prices_{cutoff.strftime('%Y%m%dT%H%M%SZ')}.parquet"
    if path.exists():
        raise FileExistsError(f"Ya existe {path}.")
    stored = frame.copy()
    stored.index = pd.to_datetime(stored.index, utc=True)
    stored = stored.reset_index(names="timestamp")
    _write_parquet(stored, path)
    return path


def cargar_historial_swaps(cache_dir: str | Path) -> tuple[pd.DataFrame, list[Path]]:
    """Combina snapshots solapados y deduplica eventos por transacción y log."""
    paths = sorted(Path(cache_dir).glob("swaps_*.parquet"))
    if not paths:
        return pd.DataFrame(), []
    frame = pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)
    keys = [key for key in ("hash_tx", "log_index") if key in frame.columns]
    if keys:
        frame = frame.drop_duplicates(keys, keep="last")
    return frame.sort_values(["timestamp", *keys]).reset_index(drop=True), paths


def cargar_historial_precios(cache_dir: str | Path) -> tuple[pd.DataFrame, list[Path]]:
    """Une velas horarias de snapshots y conserva la última versión de cada hora."""
    paths = sorted(Path(cache_dir).glob("prices_*.parquet"))
    if not paths:
        return pd.DataFrame(columns=["Close"]), []
    frame = pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    price_column = "Close" if "Close" in frame.columns else "price"
    frame = frame.drop_duplicates("timestamp", keep="last").sort_values("timestamp")
    return frame.set_index("timestamp")[[price_column]].rename(columns={price_column: "Close"}), paths


def reconstruir_perfiles_desde_cache(
    *,
    raw_cache_dir: str | Path = "data/raw/alchemy",
    output_dir: str | Path = "data/derived/alchemy_wallet_profiles",
    as_of: datetime | pd.Timestamp | str | None = None,
    lookback_hours: float = DEFAULT_LOOKBACK_HOURS,
    actualizar_precios: bool = True,
    export_png: bool = True,
) -> Path:
    """Recalcula los artefactos sin llamar a Alchemy.

    Es la vía de recuperación cuando ya se descargaron y guardaron los swaps,
    pero falló Yahoo Finance, Kaleido o la exportación del reporte. Sólo hace,
    de forma opcional, una consulta ligera a Yahoo para completar precios.
    """
    cutoff = _as_utc(as_of)
    historical, swap_paths = cargar_historial_swaps(raw_cache_dir)
    if historical.empty:
        raise FileNotFoundError(
            f"No hay snapshots swaps_*.parquet en {Path(raw_cache_dir)}; "
            "no es posible reconstruir sin volver a descargar Alchemy."
        )
    if actualizar_precios:
        current_prices = descargar_precios_horarios(as_of=cutoff, lookback_hours=lookback_hours)
        price_path = Path(raw_cache_dir) / f"prices_{cutoff.strftime('%Y%m%dT%H%M%SZ')}.parquet"
        if not price_path.exists():
            guardar_cache_precios(current_prices, raw_cache_dir, cutoff)
    prices, price_paths = cargar_historial_precios(raw_cache_dir)
    logical = canonicalizar_swaps_logicos(historical)
    cycles = construir_ciclos_hold(logical)
    ledger = construir_ledger_decisiones(logical, cycles, prices, as_of=cutoff)
    profiles = construir_perfiles_wallet(ledger, logical, cycles)
    state = construir_estado_rl(logical, cycles, profiles, as_of=cutoff)
    source_paths = list(dict.fromkeys([*swap_paths, *price_paths]))
    return exportar_snapshot_derivado(
        swaps_logicos=logical, ciclos_hold=cycles, ledger=ledger, perfiles=profiles,
        estado_rl=state, source_paths=source_paths, as_of=cutoff, output_dir=output_dir,
        lookback_hours=lookback_hours, export_png=export_png,
    )


def ejecutar_pipeline_perfiles(
    *,
    rpc_url: str,
    pool_address: str,
    lookback_hours: float = DEFAULT_LOOKBACK_HOURS,
    as_of: datetime | pd.Timestamp | str | None = None,
    raw_cache_dir: str | Path = "data/raw/alchemy",
    output_dir: str | Path = "data/derived/alchemy_wallet_profiles",
    block_span: int = DEFAULT_BLOCK_SPAN,
    pausa_seg: float = 0.05,
    verbose: bool = True,
) -> Path:
    """Ejecuta Alchemy, actualiza la historia local y exporta perfiles y gráficas."""
    cutoff = _as_utc(as_of)
    w3 = conectar_polygon(rpc_url)
    pool = obtener_contrato_pool(w3, pool_address)
    events = descargar_swaps(
        w3, pool, lookback_hours, as_of=cutoff, block_span=block_span, pausa_seg=pausa_seg, verbose=verbose
    )
    current = construir_tabla_swaps(w3, events, verbose=verbose)
    raw_path = guardar_cache_swaps(current, raw_cache_dir, cutoff)
    current_prices = descargar_precios_horarios(as_of=cutoff, lookback_hours=lookback_hours)
    price_path = guardar_cache_precios(current_prices, raw_cache_dir, cutoff)
    historical, source_paths = cargar_historial_swaps(raw_cache_dir)
    prices, price_paths = cargar_historial_precios(raw_cache_dir)
    logical = canonicalizar_swaps_logicos(historical)
    cycles = construir_ciclos_hold(logical)
    ledger = construir_ledger_decisiones(logical, cycles, prices, as_of=cutoff)
    profiles = construir_perfiles_wallet(ledger, logical, cycles)
    state = construir_estado_rl(logical, cycles, profiles, as_of=cutoff)
    source_paths = list(dict.fromkeys([*source_paths, *price_paths, raw_path, price_path]))
    return exportar_snapshot_derivado(
        swaps_logicos=logical, ciclos_hold=cycles, ledger=ledger, perfiles=profiles,
        estado_rl=state, source_paths=source_paths, as_of=cutoff, output_dir=output_dir,
        lookback_hours=lookback_hours,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Perfiles Buy/Sell/Hold de wallets POL con Alchemy.")
    parser.add_argument("--rpc-url", default=os.environ.get("ALCHEMY_RPC_URL"), help="URL RPC Alchemy; preferiblemente usa ALCHEMY_RPC_URL.")
    parser.add_argument("--pool-address", default=DEFAULT_POOL_ADDRESS, help="Pool Uniswap V3 POL/USDC.")
    parser.add_argument("--lookback-hours", type=float, default=DEFAULT_LOOKBACK_HOURS)
    parser.add_argument("--as-of", default=None, help="Corte UTC ISO-8601; por defecto ahora.")
    parser.add_argument("--raw-cache-dir", default="data/raw/alchemy")
    parser.add_argument("--output-dir", default="data/derived/alchemy_wallet_profiles")
    parser.add_argument("--block-span", type=int, default=DEFAULT_BLOCK_SPAN)
    parser.add_argument("--pause-seconds", type=float, default=0.05)
    return parser


if __name__ == "__main__":
    # ``%run`` de Colab/IPython ejecuta este archivo con argumentos del kernel.
    # No se intenta lanzar el pipeline ni se expone una clave en ese contexto.
    if "ipykernel" in sys.modules or "IPython" in sys.modules:
        print(
            "Módulo cargado. En notebook usa ejecutar_pipeline_perfiles(\n"
            "    rpc_url=os.environ['ALCHEMY_RPC_URL'],\n"
            "    pool_address=DEFAULT_POOL_ADDRESS,\n"
            ")"
        )
    else:
        args = build_parser().parse_args()
        if not args.rpc_url:
            raise SystemExit("Define ALCHEMY_RPC_URL o proporciona --rpc-url.")
        result = ejecutar_pipeline_perfiles(
            rpc_url=args.rpc_url, pool_address=args.pool_address, lookback_hours=args.lookback_hours,
            as_of=args.as_of, raw_cache_dir=args.raw_cache_dir, output_dir=args.output_dir,
            block_span=args.block_span, pausa_seg=args.pause_seconds,
        )
        print(f"Perfil exportado: {result}")
