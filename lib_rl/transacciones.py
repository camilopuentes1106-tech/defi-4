"""Módulo autónomo de transacciones: descarga y procesamiento de swaps Uniswap V3 en Polygon.

Descarga eventos Swap de pools Uniswap V3 (vía Alchemy RPC), enriquece cada transacción
con metadata on-chain (gas en POL y USDC, cantidades netas) y calcula retornos forward con yfinance.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yfinance as yf

# ──────────────────────────────────────────────────────────────────────────────
# CONSTANTES
# ──────────────────────────────────────────────────────────────────────────────

ZONA_COLOMBIA = timezone(timedelta(hours=-5))

DEFAULT_POOL_ADDRESS = "0xA374094527e1673A86dE625aa59517c5dE346d32"  # WPOL/USDC 0.05%
DEFAULT_TICKER_YF = "POL28321-USD"

INTERVALOS_YF: dict[str, timedelta] = {
    "1m": timedelta(minutes=1),
    "2m": timedelta(minutes=2),
    "5m": timedelta(minutes=5),
    "15m": timedelta(minutes=15),
    "30m": timedelta(minutes=30),
    "1h": timedelta(hours=1),
    "1d": timedelta(days=1),
}

HISTORIA_MAX_DIAS: dict[str, int] = {
    "1m": 7,
    "2m": 60,
    "5m": 60,
    "15m": 60,
    "30m": 60,
    "1h": 730,
    "1d": 99_999,
}

ABI_SWAP_UNISWAP_V3 = json.loads("""[{
    "anonymous": false,
    "inputs": [
        {"indexed": true,  "internalType": "address", "name": "sender",        "type": "address"},
        {"indexed": true,  "internalType": "address", "name": "recipient",     "type": "address"},
        {"indexed": false, "internalType": "int256",  "name": "amount0",       "type": "int256"},
        {"indexed": false, "internalType": "int256",  "name": "amount1",       "type": "int256"},
        {"indexed": false, "internalType": "uint160", "name": "sqrtPriceX96",  "type": "uint160"},
        {"indexed": false, "internalType": "uint128", "name": "liquidity",     "type": "uint128"},
        {"indexed": false, "internalType": "int24",   "name": "tick",          "type": "int24"}
    ],
    "name": "Swap",
    "type": "event"
}]""")


def _require_web3() -> tuple[Any, Any]:
    """Carga Web3 bajo demanda para no fallar en entornos locales/offline."""
    try:
        from web3 import Web3
        from web3.middleware import ExtraDataToPOAMiddleware
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Instala web3 para descargar swaps on-chain: pip install web3>=7.0") from exc
    return Web3, ExtraDataToPOAMiddleware


# ──────────────────────────────────────────────────────────────────────────────
# CLASE PRINCIPAL
# ──────────────────────────────────────────────────────────────────────────────

class SwapPipeline:
    """Pipeline completo para descargar y enriquecer swaps Uniswap V3 en Polygon.

    Parámetros
    ----------
    rpc_url : str
        Endpoint RPC de Alchemy (Polygon mainnet).
    pool_address : str
        Dirección del contrato del pool Uniswap V3.
    ticker_yf : str
        Ticker de yfinance para el token0 (ej. 'POL28321-USD').
    decimales_t0 : int
        Decimales del token0 (ej. 18 para WPOL).
    decimales_t1 : int
        Decimales del token1 (ej. 6 para USDC).
    nombre_t0 : str
        Nombre corto del token0 (ej. 'pol').
    nombre_t1 : str
        Nombre corto del token1 (ej. 'usdc').
    abi : list | None
        ABI del evento Swap.
    zona : timezone
        Zona horaria para timestamps.
    cache_path : str | None
        Ruta del archivo parquet de cache.
    """

    def __init__(
        self,
        rpc_url: str = "",
        pool_address: str = DEFAULT_POOL_ADDRESS,
        ticker_yf: str = DEFAULT_TICKER_YF,
        decimales_t0: int = 18,
        decimales_t1: int = 6,
        nombre_t0: str = "pol",
        nombre_t1: str = "usdc",
        abi: list | None = None,
        zona: timezone = ZONA_COLOMBIA,
        cache_path: str | None = None,
    ) -> None:
        self.rpc_url = rpc_url
        self.pool_address = pool_address
        self.ticker_yf = ticker_yf
        self.decimales_t0 = decimales_t0
        self.decimales_t1 = decimales_t1
        self.nombre_t0 = nombre_t0
        self.nombre_t1 = nombre_t1
        self.abi = abi or ABI_SWAP_UNISWAP_V3
        self.zona = zona
        self.cache_path = cache_path

        self.df: pd.DataFrame = pd.DataFrame()
        self._w3: Any = None
        self._pool: Any = None

    @property
    def col_cantidad_t0(self) -> str:
        return f"{self.nombre_t0}_cantidad"

    @property
    def col_cantidad_t1(self) -> str:
        return f"{self.nombre_t1}_cantidad"

    @property
    def col_precio(self) -> str:
        return "precio_ejecutado"

    def _conectar(self) -> None:
        if self._w3 is not None:
            return
        if not self.rpc_url:
            raise ValueError("Se requiere rpc_url para descargar swaps de la blockchain.")
        Web3, ExtraDataToPOAMiddleware = _require_web3()
        w3 = Web3(Web3.HTTPProvider(self.rpc_url))
        w3.middleware_onion.inject(ExtraDataToPOAMiddleware, layer=0)
        if not w3.is_connected():
            raise ConnectionError("No fue posible conectar con el endpoint RPC de Polygon.")
        self._w3 = w3
        self._pool = w3.eth.contract(
            address=w3.to_checksum_address(self.pool_address),
            abi=self.abi,
        )

    def _bloque_por_timestamp(self, ts_objetivo: int, bloque_max: int) -> int:
        lo, hi = 0, bloque_max
        while lo < hi:
            mid = (lo + hi + 1) // 2
            ts_mid = self._w3.eth.get_block(mid)["timestamp"]
            if ts_mid <= ts_objetivo:
                lo = mid
            else:
                hi = mid - 1
        return lo

    def _descargar_eventos(
        self,
        horas: float,
        paso: int,
        pausa_seg: float,
        verbose: bool,
    ) -> list:
        ahora_utc = datetime.now(timezone.utc)
        ts_inicio_dt = ahora_utc - timedelta(hours=horas)
        ts_inicio = int(ts_inicio_dt.timestamp())
        bloque_fin = self._w3.eth.block_number

        if verbose:
            print(
                f"[descarga] buscando bloque para "
                f"{ts_inicio_dt.strftime('%Y-%m-%d %H:%M:%S')} UTC..."
            )

        bloque_ini = self._bloque_por_timestamp(ts_inicio, bloque_fin)
        n_bloques = bloque_fin - bloque_ini

        if verbose:
            print(
                f"[descarga] bloques {bloque_ini} -> {bloque_fin} "
                f"({n_bloques} bloques / {horas:.2f} h)"
            )

        eventos: list[dict] = []
        total_lotes = max(1, int(np.ceil(n_bloques / paso)))
        lote_n = 0

        for inicio in range(bloque_ini, bloque_fin + 1, paso):
            fin = min(inicio + paso - 1, bloque_fin)
            try:
                batch = self._pool.events.Swap.get_logs(
                    from_block=inicio,
                    to_block=fin,
                )
                eventos.extend(batch)
            except Exception as exc:
                print(f"[error] lote {inicio}-{fin}: {exc}")

            lote_n += 1
            if verbose and (lote_n % 50 == 0 or lote_n == total_lotes):
                print(
                    f"  lote {lote_n}/{total_lotes} "
                    f"({lote_n / total_lotes * 100:.0f}%) - "
                    f"swaps: {len(eventos)}"
                )

            time.sleep(pausa_seg)

        if verbose:
            print(f"[descarga] total swaps encontrados: {len(eventos)}")

        return eventos

    def _enriquecer_evento(self, swap: dict) -> dict:
        args = swap["args"]
        n_bloque = swap["blockNumber"]
        tx_hash_val = swap["transactionHash"]
        hash_tx = tx_hash_val.hex() if hasattr(tx_hash_val, "hex") else str(tx_hash_val)

        if not hash_tx.startswith("0x"):
            hash_tx = "0x" + hash_tx

        try:
            bloque = self._w3.eth.get_block(n_bloque)
            ts_unix = bloque["timestamp"]
            ts_dt = datetime.fromtimestamp(ts_unix, self.zona)

            tx = self._w3.eth.get_transaction(hash_tx)
            recibo = self._w3.eth.get_transaction_receipt(hash_tx)

            gas_t0 = (
                recibo["gasUsed"]
                * recibo.get("effectiveGasPrice", tx.get("gasPrice", 0))
            ) / 10 ** self.decimales_t0

            wallet = tx["from"]

        except Exception as exc:
            print(f"[warn] {hash_tx[:14]}... RPC error: {exc}")
            ts_unix = 0
            ts_dt = None
            gas_t0 = np.nan
            wallet = "error"

        t0_raw = args["amount0"] / 10 ** self.decimales_t0
        t1_raw = args["amount1"] / 10 ** self.decimales_t1

        direccion = (
            "Buy" if t0_raw < 0 else
            "Sell" if t0_raw > 0 else
            "Unknown"
        )

        t0_cantidad = abs(t0_raw)
        t1_cantidad = abs(t1_raw)
        precio_exec = (
            t1_cantidad / t0_cantidad
            if t0_cantidad > 0
            else np.nan
        )

        return {
            "timestamp": ts_dt,
            "timestamp_unix": ts_unix,
            "bloque": n_bloque,
            "hash_tx": hash_tx,
            "wallet": wallet,
            "direccion": direccion,
            self.col_cantidad_t0: t0_cantidad,
            self.col_cantidad_t1: t1_cantidad,
            self.col_precio: precio_exec,
            f"gas_{self.nombre_t0}": gas_t0,
        }

    def _construir_df(self, eventos: list, verbose: bool) -> pd.DataFrame:
        filas = []
        for i, swap in enumerate(eventos, 1):
            if verbose and (i % 20 == 0 or i == len(eventos)):
                print(f"  enriqueciendo {i}/{len(eventos)}...")
            filas.append(self._enriquecer_evento(swap))

        df = pd.DataFrame(filas)
        if df.empty:
            return df

        col_gas_t0 = f"gas_{self.nombre_t0}"
        col_gas_t1 = f"gas_{self.nombre_t1}"
        col_neto_t1 = f"{self.nombre_t1}_neto"

        df[col_gas_t1] = df[col_gas_t0] * df[self.col_precio]
        df[col_neto_t1] = (df[self.col_cantidad_t1] - df[col_gas_t1]).clip(lower=0)

        cols = [
            "timestamp",
            "bloque",
            "hash_tx",
            "wallet",
            "direccion",
            self.col_cantidad_t0,
            self.col_cantidad_t1,
            self.col_precio,
            col_gas_t0,
            col_gas_t1,
            col_neto_t1,
        ]

        return df[cols].sort_values("timestamp").reset_index(drop=True)

    def _guardar_cache(self, df: pd.DataFrame) -> None:
        if self.cache_path is None:
            return
        path = Path(self.cache_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False)
        print(f"[cache] guardado -> {path} ({len(df)} swaps)")

    def _cargar_cache(self) -> pd.DataFrame | None:
        if self.cache_path is None:
            return None
        path = Path(self.cache_path)
        if not path.exists():
            return None
        df = pd.read_parquet(path)
        print(f"[cache] cargado <- {path} ({len(df)} swaps)")
        return df

    def _descargar_velas(self, intervalo: str, horas_ventana: float) -> pd.DataFrame:
        delta_forward = INTERVALOS_YF[intervalo]
        horas_forward = delta_forward.total_seconds() / 3600
        horas_necesarias = (horas_ventana + horas_forward) * 2.0
        dias_necesarios = horas_necesarias / 24
        dias_max = HISTORIA_MAX_DIAS[intervalo]
        dias_req = min(max(dias_necesarios, 1), dias_max)

        ahora = datetime.now(timezone.utc)
        inicio = (ahora - timedelta(days=dias_req)).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        fin = (ahora + timedelta(days=1)).replace(
            hour=0, minute=0, second=0, microsecond=0
        )

        velas = yf.download(
            self.ticker_yf,
            start=inicio,
            end=fin,
            interval=intervalo,
            progress=False,
            auto_adjust=True,
        )

        if velas.empty:
            return pd.DataFrame()

        if isinstance(velas.columns, pd.MultiIndex):
            velas.columns = velas.columns.get_level_values(0)

        idx = velas.index
        velas.index = (
            idx.tz_localize("UTC") if idx.tz is None
            else idx.tz_convert("UTC")
        )

        return velas[["Close"]].dropna()

    @staticmethod
    def _precio_forward(
        ts_utc: pd.Timestamp,
        delta: timedelta,
        velas: pd.DataFrame,
    ) -> float:
        objetivo = ts_utc + delta
        candidatos = velas[velas.index <= objetivo]
        if candidatos.empty:
            return np.nan
        return float(candidatos["Close"].iloc[-1])

    def ejecutar(
        self,
        horas: float = 24.0,
        horizontes: list[str] | None = None,
        forzar_descarga: bool = False,
        paso: int = 10,
        pausa_seg: float = 0.05,
        verbose: bool = True,
    ) -> pd.DataFrame:
        """Ejecuta el pipeline completo de swaps y devuelve el DataFrame procesado."""
        if horizontes is None:
            horizontes = ["1m", "2m", "5m", "15m", "30m", "1h"]

        invalidos = [iv for iv in horizontes if iv not in INTERVALOS_YF]
        if invalidos:
            raise ValueError(
                f"Intervalos no soportados: {invalidos}. "
                f"Válidos: {list(INTERVALOS_YF)}"
            )

        # 1. Cargar cache
        df = None
        if not forzar_descarga:
            df = self._cargar_cache()

        # 2. Descargar on-chain si no hay cache
        if df is None:
            self._conectar()
            eventos = self._descargar_eventos(horas, paso, pausa_seg, verbose)
            df = self._construir_df(eventos, verbose)
            if self.cache_path is not None:
                self._guardar_cache(df)

        if df is None or df.empty:
            self.df = df if df is not None else pd.DataFrame()
            return self.df

        # 3. Retornos forward si faltan
        faltan_retornos = any(f"retorno_{iv}" not in df.columns for iv in horizontes)
        if faltan_retornos:
            ts_min = pd.to_datetime(df["timestamp"], utc=True).min()
            horas_reales = (
                pd.Timestamp.now(tz="UTC") - ts_min
            ).total_seconds() / 3600
            df = self._agregar_retornos_forward(df, horizontes, horas_reales, verbose)

        self.df = df
        return self.df

    def _agregar_retornos_forward(
        self,
        df: pd.DataFrame,
        horizontes: list[str],
        horas_ventana: float,
        verbose: bool,
    ) -> pd.DataFrame:
        df = df.copy()
        df["_ts"] = pd.to_datetime(df["timestamp"], utc=True)

        for iv in horizontes:
            if verbose:
                print(f"[yfinance] descargando velas {iv}...")

            velas = self._descargar_velas(iv, horas_ventana)
            delta = INTERVALOS_YF[iv]

            col_fwd = f"precio_fwd_{iv}"
            col_retorno = f"retorno_{iv}"
            col_ganancia = f"ganancia_{self.nombre_t1}_{iv}"

            if velas.empty:
                if verbose:
                    print(f"  [warn] sin datos para {iv}")
                if col_fwd not in df.columns:
                    df[col_fwd] = np.nan
                if col_retorno not in df.columns:
                    df[col_retorno] = np.nan
                if col_ganancia not in df.columns:
                    df[col_ganancia] = np.nan
                continue

            df[col_fwd] = df["_ts"].apply(
                lambda ts: self._precio_forward(ts, delta, velas)
            )

            signo = np.where(df["direccion"] == "Buy", 1, -1)

            df[col_retorno] = (
                signo
                * (df[col_fwd] - df[self.col_precio])
                / df[self.col_precio]
            )

            df[col_ganancia] = df[col_retorno] * df[self.col_cantidad_t1]

        return df.drop(columns=["_ts"])


__all__ = [
    "DEFAULT_POOL_ADDRESS",
    "DEFAULT_TICKER_YF",
    "INTERVALOS_YF",
    "SwapPipeline",
]
