"""Pipeline reproducible de perfiles de wallets al MDP/Bellman de POL.

No descarga datos ni consulta Alchemy. Parte de un snapshot ya generado por
los módulos de perfiles y produce las tablas que pide la guía: cohorte de
wallets dirigidas, estado horario, transición empírica, política de Bellman y
una simulación/replay explicable.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .mdp import (
    BellmanResult,
    EmpiricalMDP,
    construir_mdp_empirico,
    construir_observaciones_horarias,
    replay_historico,
    resolver_bellman_finito,
    simular_politica,
    tabla_politica,
    verificar_probabilidades,
)
from ..wallets.signals import (
    CONSISTENCY_THRESHOLD_RL,
    calcular_wallets_ganadoras_1h_consistentes,
)


@dataclass(frozen=True)
class RLPipelineResult:
    """Resultado en memoria y ubicación de los artefactos reproducibles."""

    output_dir: Path
    wallets_dirigidas: pd.DataFrame
    observaciones: pd.DataFrame
    mdp: EmpiricalMDP
    bellman: BellmanResult
    politica: pd.DataFrame
    replay: pd.DataFrame


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _unique_output_dir(parent: Path, label: str) -> Path:
    parent.mkdir(parents=True, exist_ok=True)
    candidate = parent / label
    suffix = 1
    while candidate.exists():
        candidate = parent / f"{label}_{suffix:02d}"
        suffix += 1
    candidate.mkdir()
    return candidate


def _snapshot_as_of(observaciones: pd.DataFrame) -> pd.Timestamp:
    if observaciones.empty or "as_of" not in observaciones:
        return pd.Timestamp.now(tz="UTC")
    values = pd.to_datetime(observaciones["as_of"], utc=True, errors="coerce").dropna()
    return values.max() if not values.empty else pd.Timestamp.now(tz="UTC")


def ejecutar_rl_desde_snapshot(
    *,
    snapshot_dir: str | Path,
    output_dir: str | Path = "data/derived/pol_rl_bellman",
    consistency_threshold: float = CONSISTENCY_THRESHOLD_RL,
    flat_band: float = 0.001,
    profile_weight: float = 0.25,
    laplace_alpha: float = 1.0,
    horizon: int = 24,
    gamma: float = 0.99,
) -> RLPipelineResult:
    """Ejecuta el flujo completo de la guía desde un snapshot de perfiles.

    Requiere tres Parquet del snapshot: ``perfiles_wallet``,
    ``swaps_logicos`` y ``ledger_decisiones``. La cohorte está restringida a
    ganadoras dirigidas de 1 h con ``consistency_score >= 0.80`` por defecto.
    Para cada hora, el estado se recalcula sólo con decisiones que ya habían
    madurado en esa hora.
    """
    root = Path(snapshot_dir)
    sources = {
        "perfiles_wallet": root / "perfiles_wallet.parquet",
        "swaps_logicos": root / "swaps_logicos.parquet",
        "ledger_decisiones": root / "ledger_decisiones.parquet",
    }
    missing = [str(path) for path in sources.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Faltan artefactos del snapshot: " + ", ".join(missing))

    profiles = pd.read_parquet(sources["perfiles_wallet"])
    swaps = pd.read_parquet(sources["swaps_logicos"])
    ledger = pd.read_parquet(sources["ledger_decisiones"])
    directed = calcular_wallets_ganadoras_1h_consistentes(
        profiles, consistency_threshold=consistency_threshold,
    )
    observations = construir_observaciones_horarias(
        swaps, ledger, consistency_threshold=consistency_threshold, flat_band=flat_band,
    )
    if observations.empty:
        raise ValueError("No hay suficientes precios horarios para construir el MDP.")
    mdp = construir_mdp_empirico(
        observations, profile_weight=profile_weight, laplace_alpha=laplace_alpha,
    )
    bellman = resolver_bellman_finito(mdp, horizon=horizon, gamma=gamma)
    policy = tabla_politica(mdp, bellman)
    replay = replay_historico(observations, mdp, bellman)
    simulation = simular_politica(mdp, bellman)
    verification = verificar_probabilidades(mdp)

    as_of = _snapshot_as_of(observations)
    label = "snapshot_" + as_of.strftime("%Y%m%dT%H%M%SZ") + "_rl_1h"
    target = _unique_output_dir(Path(output_dir), label)
    directed.to_parquet(target / "wallets_dirigidas_1h.parquet", index=False)
    observations.to_parquet(target / "observaciones_rl_1h.parquet", index=False)
    policy.to_parquet(target / "politica_bellman.parquet", index=False)
    replay.to_parquet(target / "replay_historico.parquet", index=False)
    simulation.to_parquet(target / "simulacion_mdp.parquet", index=False)
    verification.to_parquet(target / "verificacion_transiciones.parquet", index=False)
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "as_of": as_of.isoformat(),
        "agent_frequency": "1h",
        "state": "regimen_mercado × senal_wallets × posicion",
        "actions": ["BUY_POL", "SELL_POL", "HOLD"],
        "position_model": "long_only_binary: 0=USDC, 1=POL",
        "consistency_threshold": consistency_threshold,
        "flat_band": flat_band,
        "profile_weight": profile_weight,
        "laplace_alpha": laplace_alpha,
        "bellman_horizon": horizon,
        "gamma": gamma,
        "n_wallets_dirigidas": int(directed["wallet"].nunique()),
        "n_observaciones": int(len(observations)),
        "sources": {
            name: {"path": str(path), "sha256": _hash_file(path)}
            for name, path in sources.items()
        },
    }
    (target / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8",
    )
    return RLPipelineResult(
        output_dir=target,
        wallets_dirigidas=directed,
        observaciones=observations,
        mdp=mdp,
        bellman=bellman,
        politica=policy,
        replay=replay,
    )
