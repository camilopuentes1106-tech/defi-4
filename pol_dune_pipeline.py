"""Extracción auditable de una cohorte de wallets de POL mediante Dune.

Esta fase conserva los datos fuente y una muestra reproducible. No calcula PnL
ni clasifica las wallets por desempeño.

En Colab:
    !pip install -r requirements.txt
    import os
    from getpass import getpass
    os.environ["DUNE_API_KEY"] = getpass("Dune API key: ")
    %run pol_dune_pipeline.py
    run_full_pipeline()
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable
from urllib.parse import urlencode

import pandas as pd
import requests


MIGRATION_START = pd.Timestamp("2024-09-04T00:00:00Z")
POLYGON_CHAIN = "polygon"
# Dirección del wrapper histórico del token nativo (MATIC/POL) usado en DEX.
WRAPPED_POL_ADDRESS = "0x0d500b1d8e8ef31e21c99d1db9a6444d3adf1270"
DUNE_API_BASE = "https://api.dune.com/api/v1"
SUCCESS_STATE = "QUERY_STATE_COMPLETED"
FAILURE_STATES = {"QUERY_STATE_FAILED", "QUERY_STATE_CANCELLED", "QUERY_STATE_EXPIRED"}
RETRYABLE_STATUS = {429, 500, 502, 503, 504}
RESULT_PAGE_SIZE = 1_000
DEFAULT_WALLETS_PER_QUINTILE = 2
DEFAULT_WALLETS_PER_QUERY = 100
ADDRESS_PATTERN = re.compile(r"^0x[a-f0-9]{40}$")
NODE_FETCH_SCRIPT = r"""
const chunks = [];
process.stdin.setEncoding("utf8");
process.stdin.on("data", chunk => chunks.push(chunk));
process.stdin.on("end", async () => {
  try {
    const request = JSON.parse(chunks.join(""));
    const response = await fetch(request.url, {
      method: request.method,
      headers: request.headers,
      body: request.body,
    });
    const body = await response.text();
    process.stdout.write(JSON.stringify({status: response.status, body}));
  } catch (error) {
    process.stderr.write(String(error));
    process.exitCode = 1;
  }
});
"""


@dataclass(frozen=True)
class MonthRange:
    """Intervalo semiabierto UTC [start, end) que no cruza un mes."""

    start: pd.Timestamp
    end: pd.Timestamp

    @property
    def partition(self) -> str:
        return f"year={self.start.year}/month={self.start.month:02d}"


@dataclass(frozen=True)
class ExecutionResult:
    execution_id: str
    rows: list[dict[str, Any]]
    row_count: int
    status: dict[str, Any]


class DuneApi:
    """Cliente REST de Dune; la clave nunca se persiste en disco."""

    def __init__(
        self,
        api_key: str,
        timeout_seconds: int = 60,
        transport: str = "requests",
    ) -> None:
        if not api_key:
            raise ValueError("Define DUNE_API_KEY antes de ejecutar el pipeline.")
        if transport not in {"requests", "curl", "node"}:
            raise ValueError("transport debe ser 'requests', 'curl' o 'node'.")
        self.timeout_seconds = timeout_seconds
        self.transport = transport
        self.api_key = api_key
        if transport == "requests":
            self.session = requests.Session()
            self.session.headers.update({
                "X-Dune-API-Key": api_key,
                "Content-Type": "application/json",
                "User-Agent": "pol-dune-data-pipeline/1.0",
            })
        elif transport == "curl":
            self.curl_executable = shutil.which("curl.exe") or shutil.which("curl")
            if not self.curl_executable:
                raise RuntimeError("No se encontró curl para el transporte alternativo.")
        else:
            self.node_executable = shutil.which("node.exe") or shutil.which("node")
            if not self.node_executable:
                raise RuntimeError("No se encontró Node.js para el transporte alternativo.")

    def _request(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        """Reintenta sólo límites de tasa, caídas transitorias y errores de red."""
        if self.transport == "curl":
            return self._curl_request(method, path, **kwargs)
        if self.transport == "node":
            return self._node_request(method, path, **kwargs)

        last_error: Exception | None = None
        for attempt in range(6):
            try:
                response = self.session.request(
                    method,
                    f"{DUNE_API_BASE}{path}",
                    timeout=self.timeout_seconds,
                    **kwargs,
                )
                if response.status_code in RETRYABLE_STATUS:
                    last_error = RuntimeError(f"HTTP transitorio {response.status_code}")
                    retry_after = response.headers.get("Retry-After")
                    delay = float(retry_after) if retry_after else min(30.0, 2.0**attempt)
                    time.sleep(delay)
                    continue
                response.raise_for_status()
                return response.json()
            except requests.RequestException as exc:
                last_error = exc
                if attempt < 5:
                    time.sleep(min(30.0, 2.0**attempt))
        raise RuntimeError(f"Dune no respondió correctamente: {last_error}")

    def _curl_request(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        """Transporte alternativo para entornos Windows con TLS de Python dañado."""
        params = kwargs.pop("params", None) or {}
        payload = kwargs.pop("json", None)
        if kwargs:
            raise TypeError(f"Argumentos no soportados por curl: {sorted(kwargs)}")
        url = f"{DUNE_API_BASE}{path}"
        if params:
            url = f"{url}?{urlencode(params)}"

        command = [
            self.curl_executable,
            "--silent",
            "--show-error",
            "--fail-with-body",
            "--request",
            method,
            url,
            "--header",
            f"X-Dune-API-Key: {self.api_key}",
            "--header",
            "Content-Type: application/json",
            "--header",
            "User-Agent: pol-dune-data-pipeline/1.0",
        ]
        if os.name == "nt":
            command.insert(1, "--ssl-no-revoke")
        if payload is not None:
            command.extend(["--data", json.dumps(payload, separators=(",", ":"))])

        # Algunos antivirus corporativos inyectan SSLKEYLOGFILE con un pipe
        # privado. curl/SChannel no puede abrirlo desde un subproceso y falla
        # antes de contactar Dune. Se elimina sólo para este hijo de curl.
        curl_environment = os.environ.copy()
        curl_environment.pop("SSLKEYLOGFILE", None)
        last_error: Exception | None = None
        for attempt in range(6):
            try:
                completed = subprocess.run(
                    command,
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout_seconds,
                    env=curl_environment,
                )
                if completed.returncode == 0:
                    return json.loads(completed.stdout)
                last_error = RuntimeError(completed.stderr.strip() or completed.stdout.strip())
            except (OSError, subprocess.TimeoutExpired, json.JSONDecodeError) as exc:
                last_error = exc

            if attempt < 5:
                time.sleep(min(30.0, 2.0**attempt))
        raise RuntimeError(f"curl no pudo consultar Dune: {last_error}")

    def _node_request(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        """HTTP mediante Node y certificados del sistema operativo.

        Es un respaldo para equipos Windows donde un proxy de seguridad rompe
        TLS en Python/SChannel. La petición completa se pasa por stdin, por lo
        que la clave no aparece como argumento del proceso hijo.
        """
        params = kwargs.pop("params", None) or {}
        payload = kwargs.pop("json", None)
        if kwargs:
            raise TypeError(f"Argumentos no soportados por Node: {sorted(kwargs)}")
        url = f"{DUNE_API_BASE}{path}"
        if params:
            url = f"{url}?{urlencode(params)}"
        request = {
            "url": url,
            "method": method,
            "headers": {
                "X-Dune-API-Key": self.api_key,
                "Content-Type": "application/json",
                "User-Agent": "pol-dune-data-pipeline/1.0",
            },
            "body": json.dumps(payload, separators=(",", ":")) if payload is not None else None,
        }

        last_error: Exception | None = None
        for attempt in range(6):
            try:
                completed = subprocess.run(
                    [self.node_executable, "--use-system-ca", "-e", NODE_FETCH_SCRIPT],
                    input=json.dumps(request, separators=(",", ":")),
                    check=False,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    timeout=self.timeout_seconds,
                )
                if completed.returncode != 0:
                    last_error = RuntimeError(completed.stderr.strip())
                else:
                    response = json.loads(completed.stdout)
                    status = int(response["status"])
                    if 200 <= status < 300:
                        return json.loads(response["body"])
                    last_error = RuntimeError(
                        f"Dune HTTP {status}: {response['body'][:1000]}"
                    )
                    if status not in RETRYABLE_STATUS:
                        raise last_error
            except (OSError, subprocess.TimeoutExpired, json.JSONDecodeError) as exc:
                last_error = exc

            if attempt < 5:
                time.sleep(min(30.0, 2.0**attempt))
        raise RuntimeError(f"Node no pudo consultar Dune: {last_error}")

    def execute_sql(
        self,
        sql: str,
        max_poll_seconds: int = 900,
        on_page: Callable[[str, list[dict[str, Any]], int, dict[str, Any]], None] | None = None,
    ) -> ExecutionResult:
        started = self._request("POST", "/sql/execute", json={"sql": sql})
        execution_id = started.get("execution_id")
        if not execution_id:
            raise RuntimeError(f"Dune no devolvió execution_id: {started}")

        deadline, delay = time.monotonic() + max_poll_seconds, 2.0
        status: dict[str, Any] = {}
        while time.monotonic() < deadline:
            status = self._request("GET", f"/execution/{execution_id}/status")
            state = status.get("state")
            if state == SUCCESS_STATE:
                break
            if state in FAILURE_STATES:
                raise RuntimeError(
                    f"Consulta Dune {execution_id} terminó en {state}: {status.get('error')}"
                )
            time.sleep(delay)
            delay = min(15.0, delay * 1.5)
        else:
            raise TimeoutError(f"Dune no completó {execution_id} en {max_poll_seconds} segundos.")

        rows: list[dict[str, Any]] = []
        row_count, offset, page_number = 0, 0, 0
        while True:
            result = self._request(
                "GET",
                f"/execution/{execution_id}/results",
                params={"limit": RESULT_PAGE_SIZE, "offset": offset},
            )
            page_rows = result.get("result", {}).get("rows", [])
            page_number += 1
            row_count += len(page_rows)
            if on_page is None:
                rows.extend(page_rows)
            else:
                on_page(execution_id, page_rows, page_number, result)
            next_offset = result.get("next_offset")
            if next_offset is None:
                return ExecutionResult(execution_id, rows, row_count, status)
            offset = int(next_offset)


def parse_utc(value: str | pd.Timestamp) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")
    return timestamp.floor("s")


def month_ranges(start: pd.Timestamp, end: pd.Timestamp) -> Iterable[MonthRange]:
    """Partir por mes limita tamaño, crédito consumido y alcance de reintentos."""
    if end <= start:
        raise ValueError("La fecha final debe ser posterior a la inicial.")
    cursor = start
    while cursor < end:
        next_month = (cursor + pd.offsets.MonthBegin(1)).normalize()
        chunk_end = min(next_month, end)
        yield MonthRange(cursor, chunk_end)
        cursor = chunk_end


def _timestamp(value: pd.Timestamp) -> str:
    return value.strftime("%Y-%m-%d %H:%M:%S")


def _month(value: pd.Timestamp) -> str:
    return value.strftime("%Y-%m-01")


def _time_filter(period: MonthRange, column: str) -> str:
    return (
        f"{column} >= TIMESTAMP '{_timestamp(period.start)}'\n"
        f"  AND {column} < TIMESTAMP '{_timestamp(period.end)}'"
    )


def _month_filter(period: MonthRange, alias: str) -> str:
    next_month = (period.start + pd.offsets.MonthBegin(1)).normalize()
    return (
        f"{alias}.block_month >= DATE '{_month(period.start)}'\n"
        f"  AND {alias}.block_month < DATE '{_month(next_month)}'"
    )


def _wallet_filter(wallets: list[str], alias: str) -> str:
    if not wallets:
        raise ValueError("Se requiere al menos una wallet para extraer su actividad.")
    invalid = [wallet for wallet in wallets if not ADDRESS_PATTERN.fullmatch(wallet.lower())]
    if invalid:
        raise ValueError(f"Wallets con formato inválido: {invalid[:3]}")
    return f"{alias}.tx_from IN ({','.join(wallet.lower() for wallet in wallets)})"


def candidate_wallets_sql(period: MonthRange, wallets_per_quintile: int) -> str:
    """Muestra estable de wallets sin sumar por error los hops de una ruta."""
    if wallets_per_quintile < 1:
        raise ValueError("wallets_per_quintile debe ser al menos 1.")
    return f"""
WITH wallet_transactions AS (
    SELECT
        lower(concat('0x', to_hex(d.tx_from))) AS wallet,
        date_trunc('month', d.block_time) AS month,
        d.tx_hash,
        max(coalesce(d.amount_usd, 0)) AS tx_volume_proxy_usd
    FROM dex.trades AS d
    WHERE d.blockchain = '{POLYGON_CHAIN}'
      AND {_month_filter(period, 'd')}
      AND {_time_filter(period, 'd.block_time')}
      AND (
          d.token_bought_address = {WRAPPED_POL_ADDRESS}
          OR d.token_sold_address = {WRAPPED_POL_ADDRESS}
      )
    GROUP BY 1, 2, 3
),
wallet_month AS (
    SELECT
        wallet,
        month,
        count(*) AS swap_transactions,
        sum(tx_volume_proxy_usd) AS volume_proxy_usd
    FROM wallet_transactions
    GROUP BY 1, 2
),
bucketed AS (
    SELECT
        *,
        ntile(5) OVER (PARTITION BY month ORDER BY volume_proxy_usd) AS volume_quintile
    FROM wallet_month
),
sampled AS (
    SELECT
        *,
        row_number() OVER (
            PARTITION BY month, volume_quintile
            ORDER BY wallet
        ) AS sample_rank
    FROM bucketed
)
SELECT wallet, month, swap_transactions, volume_proxy_usd, volume_quintile, sample_rank
FROM sampled
WHERE sample_rank <= {int(wallets_per_quintile)}
ORDER BY month, volume_quintile, sample_rank
""".strip()


def dex_trades_sql(period: MonthRange, wallets: list[str] | None = None) -> str:
    """Eventos DEX crudos por hop; no se agregan ni se etiquetan como buy/sell."""
    return f"""
SELECT
    d.block_time, d.block_number, d.tx_hash, d.evt_index, d.tx_from, d.tx_to,
    d.taker, d.maker, d.project, d.version AS project_version,
    d.project_contract_address,
    d.token_bought_address, d.token_bought_symbol, d.token_bought_amount,
    d.token_bought_amount_raw, d.token_sold_address, d.token_sold_symbol,
    d.token_sold_amount, d.token_sold_amount_raw, d.amount_usd
FROM dex.trades AS d
WHERE d.blockchain = '{POLYGON_CHAIN}'
  AND {_month_filter(period, 'd')}
  AND {_time_filter(period, 'd.block_time')}
  AND (
      d.token_bought_address = {WRAPPED_POL_ADDRESS}
      OR d.token_sold_address = {WRAPPED_POL_ADDRESS}
  )
  {f"AND {_wallet_filter(wallets, 'd')}" if wallets else ''}
ORDER BY d.block_time, d.block_number, d.tx_hash, d.evt_index
""".strip()


def gas_fees_sql(period: MonthRange, wallets: list[str] | None = None) -> str:
    """Un coste por hash de transacción de POL, incluso si tuvo varios hops."""
    return f"""
WITH pol_trade_transactions AS (
    SELECT DISTINCT d.tx_hash
    FROM dex.trades AS d
    WHERE d.blockchain = '{POLYGON_CHAIN}'
      AND {_month_filter(period, 'd')}
      AND {_time_filter(period, 'd.block_time')}
      AND (
          d.token_bought_address = {WRAPPED_POL_ADDRESS}
          OR d.token_sold_address = {WRAPPED_POL_ADDRESS}
      )
      {f"AND {_wallet_filter(wallets, 'd')}" if wallets else ''}
)
SELECT
    g.block_time, g.block_number, g.tx_hash, g.tx_from, g.tx_to,
    g.gas_price, g.gas_used, g.gas_limit, g.gas_limit_usage,
    g.currency_symbol, g.tx_fee, g.tx_fee_usd, g.tx_fee_raw
FROM gas.fees AS g
INNER JOIN pol_trade_transactions AS p ON p.tx_hash = g.tx_hash
WHERE g.blockchain = '{POLYGON_CHAIN}'
  AND {_month_filter(period, 'g')}
  AND {_time_filter(period, 'g.block_time')}
ORDER BY g.block_time, g.block_number, g.tx_hash
""".strip()


def prices_hour_sql(period: MonthRange) -> str:
    return f"""
SELECT p.timestamp, p.contract_address, p.symbol, p.decimals, p.price, p.volume, p.source
FROM prices.hour AS p
WHERE p.blockchain = '{POLYGON_CHAIN}'
  AND p.contract_address = {WRAPPED_POL_ADDRESS}
  AND {_time_filter(period, 'p.timestamp')}
ORDER BY p.timestamp
""".strip()


EXPECTED_COLUMNS = {
    "wallet_candidates": {
        "wallet", "month", "swap_transactions", "volume_proxy_usd",
        "volume_quintile", "sample_rank",
    },
    "dex_trades": {"block_time", "tx_hash", "evt_index", "tx_from", "amount_usd"},
    "gas_fees": {"block_time", "tx_hash", "gas_used", "tx_fee", "tx_fee_usd"},
    "prices_hour": {"timestamp", "contract_address", "price", "source"},
}
BUSINESS_KEYS = {
    "wallet_candidates": ["month", "volume_quintile", "sample_rank"],
    "dex_trades": ["tx_hash", "evt_index"],
    "gas_fees": ["tx_hash"],
    "prices_hour": ["timestamp"],
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_parquet(frame: pd.DataFrame, destination: Path) -> None:
    """Escritura atómica: nunca confundir un archivo parcial con datos válidos."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".partial")
    frame.to_parquet(temporary, index=False, engine="pyarrow")
    os.replace(temporary, destination)


def write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    temporary = path.with_suffix(".partial")
    temporary.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def initial_manifest(start: pd.Timestamp, end: pd.Timestamp, snapshot: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "pipeline": "pol-dune-data-pipeline",
        "status": "running",
        "snapshot": snapshot,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": "Dune API",
        "scope": {
            "blockchain": POLYGON_CHAIN,
            "asset_identifier": WRAPPED_POL_ADDRESS,
            "asset_note": "Wrapper histórico de MATIC/POL usado por pools DEX de Polygon.",
            "start_utc": start.isoformat(),
            "end_exclusive_utc": end.isoformat(),
        },
        "semantics": {
            "wallet_candidates": "Cohorte mensual determinista estratificada por quintil de volumen proxy DEX; no mide rentabilidad.",
            "dex_trades": "Eventos DEX crudos por hop; no equivalen aún a compras/ventas lógicas.",
            "gas_fees": "Una fila por transacción que contiene al menos un hop de POL.",
            "prices_hour": "Precio horario publicado por Dune para contrato y cadena indicados.",
        },
        "artifacts": [],
    }


def save_page_artifact(
    root: Path,
    manifest: dict[str, Any],
    dataset: str,
    period: MonthRange,
    sql: str,
    execution_id: str,
    page_status: dict[str, Any],
    page_rows: list[dict[str, Any]],
    page_number: int,
    batch_number: int | None = None,
) -> int:
    """Persiste cada página recibida sin acumular un mes entero en memoria."""
    frame = pd.DataFrame(page_rows)
    if frame.empty:
        frame = pd.DataFrame(columns=sorted(EXPECTED_COLUMNS[dataset]))
    missing = EXPECTED_COLUMNS[dataset].difference(frame.columns)
    if missing:
        raise RuntimeError(f"Esquema inesperado en {dataset}: faltan {sorted(missing)}")

    artifact_directory = root / dataset / period.partition
    if batch_number is not None:
        artifact_directory /= f"wallet_batch={batch_number:03d}"
    destination = artifact_directory / f"part-{page_number:06d}.parquet"
    write_parquet(frame, destination)
    key = BUSINESS_KEYS[dataset]
    duplicates = int(frame.duplicated(key).sum()) if not frame.empty else 0
    manifest["artifacts"].append({
        "dataset": dataset,
        "partition": period.partition,
        "wallet_batch": batch_number,
        "path": destination.relative_to(root).as_posix(),
        "rows": int(len(frame)),
        "columns": list(frame.columns),
        "dtypes": {column: str(dtype) for column, dtype in frame.dtypes.items()},
        "business_key": key,
        "duplicate_business_keys": duplicates,
        "execution_id": execution_id,
        "execution_finished_at": page_status.get("execution_ended_at"),
        "sql_sha256": hashlib.sha256(sql.encode("utf-8")).hexdigest(),
        "sql": sql,
        "file_sha256": sha256_file(destination),
    })
    return len(frame)


def wallet_batches(wallets: list[str], size: int) -> Iterable[list[str]]:
    """Partir la cohorte evita listas IN excesivas y facilita reintentos."""
    if size < 1:
        raise ValueError("wallets_per_query debe ser al menos 1.")
    for offset in range(0, len(wallets), size):
        yield wallets[offset:offset + size]


def selected_wallets(rows: list[dict[str, Any]]) -> list[str]:
    """Normaliza el resultado de Dune antes de interpolarlo como literales SQL."""
    wallets = sorted({
        str(row["wallet"]).lower()
        for row in rows
        if row.get("wallet") is not None and ADDRESS_PATTERN.fullmatch(str(row["wallet"]).lower())
    })
    if len(wallets) != len({str(row["wallet"]).lower() for row in rows if row.get("wallet") is not None}):
        raise RuntimeError("Dune devolviÃ³ una wallet candidata con formato inesperado.")
    return wallets


def run_full_pipeline(
    start: str | pd.Timestamp = MIGRATION_START,
    end: str | pd.Timestamp | None = None,
    output_dir: str | Path = "data/raw/dune",
    max_poll_seconds: int = 900,
    transport: str = "requests",
    wallets_per_quintile: int = DEFAULT_WALLETS_PER_QUINTILE,
    wallets_per_query: int = DEFAULT_WALLETS_PER_QUERY,
) -> Path:
    """Descarga una cohorte mensual estratificada y devuelve su snapshot.

    Primero se seleccionan hasta ``wallets_per_quintile`` wallets de cada uno
    de cinco quintiles de volumen por mes. Después se descargan swaps y gas de
    esas wallets en ese mismo mes; el precio horario permanece global.
    """
    start_at = parse_utc(start)
    end_at = parse_utc(end or pd.Timestamp.now(tz="UTC"))
    if start_at < MIGRATION_START:
        raise ValueError(f"El inicio debe ser >= {MIGRATION_START.date()}.")
    if wallets_per_quintile < 1:
        raise ValueError("wallets_per_quintile debe ser al menos 1.")
    if wallets_per_query < 1:
        raise ValueError("wallets_per_query debe ser al menos 1.")

    # El nombre depende del instante de extracción, no de la fecha de corte:
    # así una ejecución interrumpida no bloquea una repetición equivalente.
    snapshot = datetime.now(timezone.utc).strftime("snapshot_%Y%m%dT%H%M%SZ")
    root = Path(output_dir) / snapshot
    if root.exists():
        raise FileExistsError(f"Ya existe {root}; no se sobrescriben snapshots.")
    dune = DuneApi(os.environ.get("DUNE_API_KEY", ""), transport=transport)
    root.mkdir(parents=True)
    manifest_path = root / "manifest.json"
    manifest = initial_manifest(start_at, end_at, snapshot)
    manifest["scope"]["wallet_sampling"] = {
        "method": "Muestra determinista por mes y quintil de volumen proxy DEX.",
        "quintiles": 5,
        "wallets_per_quintile": wallets_per_quintile,
        "max_wallets_per_month": 5 * wallets_per_quintile,
        "wallet_activity_scope": "Sólo actividad de cada wallet seleccionada en su mismo mes de selección.",
    }
    write_manifest(manifest_path, manifest)

    intervals = list(month_ranges(start_at, end_at))

    def record_execution_completion(execution: ExecutionResult) -> None:
        """Añade el estado final de Dune a todas las páginas de la ejecución."""
        for artifact in manifest["artifacts"]:
            if artifact["execution_id"] == execution.execution_id:
                artifact["execution_state"] = execution.status.get("state")
                artifact["execution_finished_at"] = execution.status.get(
                    "execution_ended_at"
                )
        write_manifest(manifest_path, manifest)

    def execute_and_persist(
        dataset: str,
        period: MonthRange,
        sql: str,
        batch_number: int | None = None,
    ) -> ExecutionResult:
        """Ejecuta Dune y confirma cada página en el manifiesto de inmediato."""
        pages_saved = 0

        def persist_page(
            execution_id: str,
            page_rows: list[dict[str, Any]],
            page_number: int,
            page_status: dict[str, Any],
        ) -> None:
            nonlocal pages_saved
            pages_saved += 1
            save_page_artifact(
                root,
                manifest,
                dataset,
                period,
                sql,
                execution_id,
                page_status,
                page_rows,
                page_number,
                batch_number,
            )
            write_manifest(manifest_path, manifest)

        execution = dune.execute_sql(
            sql, max_poll_seconds=max_poll_seconds, on_page=persist_page
        )
        if pages_saved == 0:
            persist_page(execution.execution_id, [], 1, execution.status)
        record_execution_completion(execution)
        label = f" {batch_number}" if batch_number is not None else ""
        print(f"  {dataset}{label}: {execution.row_count:,} filas en {pages_saved} archivo(s)")
        return execution

    try:
        for index, period in enumerate(intervals, start=1):
            print(f"[{index}/{len(intervals)}] {period.partition}: {period.start.date()} a {period.end.date()}")
            candidates_query = candidate_wallets_sql(period, wallets_per_quintile)
            candidate_execution = dune.execute_sql(
                candidates_query, max_poll_seconds=max_poll_seconds
            )
            save_page_artifact(
                root,
                manifest,
                "wallet_candidates",
                period,
                candidates_query,
                candidate_execution.execution_id,
                candidate_execution.status,
                candidate_execution.rows,
                1,
            )
            record_execution_completion(candidate_execution)
            wallets = selected_wallets(candidate_execution.rows)
            print(f"  wallet_candidates: {len(wallets)} wallets seleccionadas")

            for batch_number, batch in enumerate(wallet_batches(wallets, wallets_per_query), start=1):
                execute_and_persist(
                    "dex_trades", period, dex_trades_sql(period, batch), batch_number
                )
                execute_and_persist(
                    "gas_fees", period, gas_fees_sql(period, batch), batch_number
                )
            if not wallets:
                print("  No hubo wallets candidatas; se conserva el precio horario del mes.")
            execute_and_persist("prices_hour", period, prices_hour_sql(period))
    except Exception:
        manifest["status"] = "failed"
        manifest["failed_at_utc"] = datetime.now(timezone.utc).isoformat()
        write_manifest(manifest_path, manifest)
        raise

    manifest["status"] = "completed"
    manifest["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    write_manifest(manifest_path, manifest)
    print(f"Extracción completa: {root}")
    return root


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extrae datos POL de Dune.")
    parser.add_argument("--start", default=MIGRATION_START.isoformat())
    parser.add_argument("--end", default=None, help="Fin exclusivo UTC; por defecto, ahora.")
    parser.add_argument("--output-dir", default="data/raw/dune")
    parser.add_argument("--max-poll-seconds", type=int, default=900)
    parser.add_argument(
        "--wallets-per-quintile",
        type=int,
        default=DEFAULT_WALLETS_PER_QUINTILE,
        help="Máximo de wallets seleccionadas en cada uno de cinco quintiles mensuales.",
    )
    parser.add_argument(
        "--wallets-per-query",
        type=int,
        default=DEFAULT_WALLETS_PER_QUERY,
        help="Máximo de wallets por consulta de swaps y gas.",
    )
    parser.add_argument(
        "--transport",
        choices=("requests", "curl", "node"),
        default="requests",
        help="Usa 'node' si el TLS de Python falla en Windows.",
    )
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    run_full_pipeline(
        start=args.start,
        end=args.end,
        output_dir=args.output_dir,
        max_poll_seconds=args.max_poll_seconds,
        transport=args.transport,
        wallets_per_quintile=args.wallets_per_quintile,
        wallets_per_query=args.wallets_per_query,
    )
