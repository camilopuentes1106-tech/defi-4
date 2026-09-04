"""Módulo autónomo de MDP empírico y solución Bellman para trading de POL.

Combina régimen de mercado con señales de wallets ganadoras en un espacio de 24 estados,
estima probabilidades de transición empíricas con suavizado de Laplace, calcula recompensas
con ventaja neta ajustada por gas y resuelve la política óptima mediante inducción hacia atrás.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

try:
    from .wallets import (
        CONSISTENCY_THRESHOLD_RL,
        calcular_wallets_ganadoras_1h_consistentes,
        construir_senales_wallets_horarias,
    )
except ImportError:
    from wallets import (
        CONSISTENCY_THRESHOLD_RL,
        calcular_wallets_ganadoras_1h_consistentes,
        construir_senales_wallets_horarias,
    )

# ──────────────────────────────────────────────────────────────────────────────
# CONSTANTES Y TIPOS
# ──────────────────────────────────────────────────────────────────────────────

ACCIONES = ("BUY_POL", "SELL_POL", "HOLD")
REGIMENES_MERCADO = ("DOWN", "FLAT", "UP")
SENALES_WALLETS = ("BUY", "SELL", "HOLD", "NEUTRAL")

State = tuple[str, str, int]

_SUPPORT_KEYS = {
    "BUY_POL": "support_buy",
    "SELL_POL": "support_sell",
    "HOLD": "support_hold",
}


def etiqueta_estado(state: State) -> str:
    """Representación legible de un estado del MDP."""
    regime, signal, position = state
    return f"{regime} | {signal} | posición={position}"


def estados_mdp() -> tuple[State, ...]:
    """Espacio de 24 estados interpretables: régimen × señal × posición."""
    return tuple(product(REGIMENES_MERCADO, SENALES_WALLETS, (0, 1)))


# ──────────────────────────────────────────────────────────────────────────────
# DATACLASSES
# ──────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class EmpiricalMDP:
    """Componentes tabulares del MDP estimado desde el historial POL."""

    states: tuple[State, ...]
    transitions: Mapping[str, np.ndarray]
    rewards: Mapping[str, np.ndarray]
    gas_cost_ratio: float
    profile_weight: float


@dataclass(frozen=True)
class BellmanResult:
    """Valores V, Q y política óptima de un horizonte finito."""

    values: np.ndarray
    q_values: np.ndarray
    policy: np.ndarray


@dataclass(frozen=True)
class RLPipelineResult:
    """Resultado en memoria de la ejecución del agente Bellman."""

    wallets_dirigidas: pd.DataFrame
    observaciones: pd.DataFrame
    mdp: EmpiricalMDP
    bellman: BellmanResult
    politica: pd.DataFrame
    replay: pd.DataFrame
    output_dir: Path | None = None


# ──────────────────────────────────────────────────────────────────────────────
# FUNCIONES DE RECOMPENSA Y ACCIONES ADMISIBLES
# ──────────────────────────────────────────────────────────────────────────────

def acciones_admisibles(posicion: int) -> tuple[str, str]:
    """Acciones válidas para un portafolio long-only binario."""
    if posicion == 0:
        return "BUY_POL", "HOLD"
    if posicion == 1:
        return "SELL_POL", "HOLD"
    raise ValueError("posicion debe ser 0 (USDC) o 1 (POL).")


def posicion_despues(posicion: int, accion: str) -> int:
    """Aplica la acción al inventario, validando admisibilidad."""
    if accion not in acciones_admisibles(posicion):
        raise ValueError(f"{accion} no es admisible para posicion={posicion}.")
    if accion == "BUY_POL":
        return 1
    if accion == "SELL_POL":
        return 0
    return posicion


def _support(accion: str, support_wallets: Mapping[str, float | int]) -> float:
    key = _SUPPORT_KEYS.get(accion, "")
    value = float(support_wallets.get(key, 0.0))
    return max(value, 0.0) if np.isfinite(value) else 0.0


def confianza_relativa_accion(
    accion: str,
    support_wallets: Mapping[str, float | int],
) -> float:
    """Confianza normalizada 0..1 si la acción domina a las otras direcciones."""
    if accion not in ACCIONES:
        raise ValueError(f"Acción no soportada: {accion}.")
    supports = {cand: _support(cand, support_wallets) for cand in ACCIONES}
    leader = max(supports, key=supports.get)
    if supports[leader] <= 0 or accion != leader:
        return 0.0
    if sum(np.isclose(v, supports[leader]) for v in supports.values()) > 1:
        return 0.0
    next_best = max(v for cand, v in supports.items() if cand != accion)
    return float((supports[accion] - next_best) / (sum(supports.values()) + 1e-12))


def desglose_recompensas(
    *,
    posicion: int,
    precio_t: float,
    precio_t1: float,
    costo_gas_ratio: float,
    support_wallets: Mapping[str, float | int] | None = None,
    profile_weight: float = 0.25,
) -> dict[str, dict[str, float]]:
    """Calcula base, ventaja y refuerzo de perfiles para las acciones válidas."""
    support_wallets = support_wallets or {}
    return_pol = precio_t1 / precio_t - 1.0
    base: dict[str, float] = {}
    for accion in acciones_admisibles(posicion):
        next_pos = posicion_despues(posicion, accion)
        cost = costo_gas_ratio if accion in {"BUY_POL", "SELL_POL"} else 0.0
        base[accion] = next_pos * return_pol - cost

    result: dict[str, dict[str, float]] = {}
    for accion, reward_base in base.items():
        alternatives = [v for cand, v in base.items() if cand != accion]
        advantage = reward_base - float(np.mean(alternatives))
        confidence = confianza_relativa_accion(accion, support_wallets)
        profile_bonus = profile_weight * confidence * advantage
        result[accion] = {
            "reward_base": float(reward_base),
            "advantage": float(advantage),
            "wallet_confidence": confidence,
            "profile_bonus": float(profile_bonus),
            "reward_final": float(reward_base + profile_bonus),
        }
    return result


def calcular_recompensa_con_perfiles(
    *,
    posicion: int,
    accion: str,
    precio_t: float,
    precio_t1: float,
    costo_gas_ratio: float,
    support_wallets: Mapping[str, float | int] | None = None,
    profile_weight: float = 0.25,
) -> dict[str, float]:
    """Devuelve el desglose de una acción particular del MDP."""
    rewards = desglose_recompensas(
        posicion=posicion, precio_t=precio_t, precio_t1=precio_t1,
        costo_gas_ratio=costo_gas_ratio, support_wallets=support_wallets,
        profile_weight=profile_weight,
    )
    if accion not in rewards:
        raise ValueError(f"{accion} no es admisible para posicion={posicion}.")
    return rewards[accion]


# ──────────────────────────────────────────────────────────────────────────────
# OBSERVACIONES HORARIAS Y CONSTRUCCIÓN DEL MDP
# ──────────────────────────────────────────────────────────────────────────────

def _price_events(swaps_logicos: pd.DataFrame) -> pd.Series:
    frame = swaps_logicos.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    frame["precio_ejecutado"] = pd.to_numeric(frame["precio_ejecutado"], errors="coerce")
    dir_col = "accion" if "accion" in frame.columns else "direccion"
    frame = frame[
        frame["timestamp"].notna()
        & (frame["precio_ejecutado"] > 0)
        & frame[dir_col].isin(["BUY_POL", "SELL_POL", "Buy", "Sell"])
    ].sort_values("timestamp")
    if frame.empty:
        return pd.Series(dtype=float, name="price")
    prices = frame.set_index("timestamp")["precio_ejecutado"]
    return prices[~prices.index.duplicated(keep="last")].rename("price")


def _last_price_before(prices: pd.Series, timestamps: pd.DatetimeIndex) -> pd.Series:
    source = prices.dropna().sort_index().rename("price").rename_axis("price_time").reset_index()
    target = pd.DataFrame({"timestamp": timestamps}).sort_values("timestamp")
    merged = pd.merge_asof(target, source, left_on="timestamp", right_on="price_time", direction="backward")
    return merged.set_index("timestamp")["price"].reindex(timestamps)


def _gas_cost_ratio(swaps_logicos: pd.DataFrame) -> float:
    notional_col = "usdc_cantidad" if "usdc_cantidad" in swaps_logicos.columns else "notional_usdc"
    gas_col = "gas_usdc"
    if not {gas_col, notional_col}.issubset(swaps_logicos.columns):
        return 0.0
    notional = pd.to_numeric(swaps_logicos[notional_col], errors="coerce")
    gas = pd.to_numeric(swaps_logicos[gas_col], errors="coerce")
    ratios = (gas / notional).replace([np.inf, -np.inf], np.nan).dropna()
    return float(ratios.median()) if not ratios.empty else 0.0


def _clasificar_regimen(momentum: pd.Series, *, flat_band: float) -> tuple[pd.Series, float, float]:
    lower, upper = -float(flat_band), float(flat_band)
    regime = pd.Series(
        np.select([momentum < lower, momentum > upper], ["DOWN", "UP"], default="FLAT"),
        index=momentum.index,
        dtype=object,
    )
    regime[momentum.isna()] = "FLAT"
    return regime, float(lower), float(upper)


def construir_observaciones_horarias(
    swaps_logicos: pd.DataFrame,
    ledger: pd.DataFrame,
    *,
    consistency_threshold: float = CONSISTENCY_THRESHOLD_RL,
    flat_band: float = 0.001,
) -> pd.DataFrame:
    """Construye las 24 observaciones horarias: mercado, señal de wallets y confianza."""
    prices = _price_events(swaps_logicos)
    if prices.empty:
        return pd.DataFrame()
    first = prices.index.min().ceil("h")
    last = prices.index.max().floor("h")
    cuts = pd.date_range(first, last, freq="h", tz="UTC")
    if len(cuts) < 2:
        return pd.DataFrame()

    current = _last_price_before(prices, cuts)
    next_hour = _last_price_before(prices, cuts + pd.Timedelta(hours=1))
    frame = pd.DataFrame({"as_of": cuts, "precio_t": current.to_numpy(), "precio_t1": next_hour.to_numpy()})

    for horizon in ("1m", "5m", "15m", "1h"):
        delta = pd.Timedelta(horizon)
        previous = _last_price_before(prices, cuts - delta)
        frame[f"retorno_{horizon}"] = frame["precio_t"].to_numpy() / previous.to_numpy() - 1.0

    frame = frame.dropna(subset=["precio_t", "precio_t1"]).copy()
    frame["momentum_compuesto"] = frame[["retorno_1m", "retorno_5m", "retorno_15m", "retorno_1h"]].mean(axis=1)
    frame["regimen_mercado"], lower, upper = _clasificar_regimen(frame["momentum_compuesto"], flat_band=flat_band)

    signals = construir_senales_wallets_horarias(ledger, frame["as_of"], consistency_threshold=consistency_threshold)
    frame = frame.merge(signals, on="as_of", how="left")
    frame["senal_wallets"] = frame["senal_wallets"].fillna("NEUTRAL")
    for col in ("confianza_wallets", "support_buy", "support_sell", "support_hold"):
        frame[col] = pd.to_numeric(frame[col], errors="coerce").fillna(0.0)

    frame["costo_gas_ratio"] = _gas_cost_ratio(swaps_logicos)
    frame.attrs["momentum_lower_threshold"] = lower
    frame.attrs["momentum_upper_threshold"] = upper
    return frame.reset_index(drop=True)


def _support_from_row(row: pd.Series) -> dict[str, float]:
    return {
        "support_buy": float(row.get("support_buy", 0.0)),
        "support_sell": float(row.get("support_sell", 0.0)),
        "support_hold": float(row.get("support_hold", 0.0)),
    }


def construir_mdp_empirico(
    observaciones: pd.DataFrame,
    *,
    profile_weight: float = 0.25,
    laplace_alpha: float = 1.0,
) -> EmpiricalMDP:
    """Estima transiciones empíricas P(s'|s,a) y recompensas esperadas para los 24 estados."""
    if laplace_alpha <= 0:
        raise ValueError("laplace_alpha debe ser > 0.")
    frame = observaciones.copy().reset_index(drop=True)
    frame = frame[
        frame["regimen_mercado"].isin(REGIMENES_MERCADO)
        & frame["senal_wallets"].isin(SENALES_WALLETS)
    ].copy()
    if len(frame) < 2:
        raise ValueError("Se requieren al menos dos observaciones horarias.")

    economic_states = tuple(product(REGIMENES_MERCADO, SENALES_WALLETS))
    economic_index = {state: idx for idx, state in enumerate(economic_states)}
    states = estados_mdp()
    state_index = {state: idx for idx, state in enumerate(states)}

    counts = np.full((len(economic_states), len(economic_states)), laplace_alpha, dtype=float)
    for i in range(len(frame) - 1):
        src = (frame.loc[i, "regimen_mercado"], frame.loc[i, "senal_wallets"])
        tgt = (frame.loc[i + 1, "regimen_mercado"], frame.loc[i + 1, "senal_wallets"])
        counts[economic_index[src], economic_index[tgt]] += 1.0
    economic_transition = counts / counts.sum(axis=1, keepdims=True)

    transitions = {action: np.zeros((len(states), len(states)), dtype=float) for action in ACCIONES}
    for src_idx, (regime, signal, pos) in enumerate(states):
        src_econ = economic_index[(regime, signal)]
        for action in acciones_admisibles(pos):
            next_pos = posicion_despues(pos, action)
            for tgt_econ, prob in zip(economic_states, economic_transition[src_econ]):
                tgt_idx = state_index[(*tgt_econ, next_pos)]
                transitions[action][src_idx, tgt_idx] = prob

    rewards = {action: np.full(len(states), np.nan, dtype=float) for action in ACCIONES}
    reward_samples: dict[tuple[tuple[str, str], int, str], list[float]] = {}
    global_samples: dict[tuple[int, str], list[float]] = {}
    for _, row in frame.iterrows():
        econ = (row["regimen_mercado"], row["senal_wallets"])
        for pos in (0, 1):
            for action in acciones_admisibles(pos):
                reward = calcular_recompensa_con_perfiles(
                    posicion=pos, accion=action, precio_t=float(row["precio_t"]),
                    precio_t1=float(row["precio_t1"]), costo_gas_ratio=float(row["costo_gas_ratio"]),
                    support_wallets=_support_from_row(row), profile_weight=profile_weight,
                )["reward_final"]
                reward_samples.setdefault((econ, pos, action), []).append(reward)
                global_samples.setdefault((pos, action), []).append(reward)

    for state_i, (regime, signal, pos) in enumerate(states):
        econ = (regime, signal)
        for action in acciones_admisibles(pos):
            samples = reward_samples.get((econ, pos, action), global_samples[(pos, action)])
            rewards[action][state_i] = float(np.mean(samples))

    return EmpiricalMDP(
        states=states,
        transitions=transitions,
        rewards=rewards,
        gas_cost_ratio=float(frame["costo_gas_ratio"].median()),
        profile_weight=float(profile_weight),
    )


def verificar_probabilidades(mdp: EmpiricalMDP, *, atol: float = 1e-10) -> pd.DataFrame:
    """Verifica que para cada estado y acción válida, la fila de probabilidades sume 1.0."""
    rows = []
    for idx, state in enumerate(mdp.states):
        for action in acciones_admisibles(state[2]):
            total = float(mdp.transitions[action][idx].sum())
            rows.append({
                "estado": etiqueta_estado(state),
                "accion": action,
                "suma_probabilidades": total,
                "valida": bool(np.isclose(total, 1.0, atol=atol)),
            })
    return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────────────────────
# SOLUCIÓN BELLMAN Y REPLAY
# ──────────────────────────────────────────────────────────────────────────────

def resolver_bellman_finito(
    mdp: EmpiricalMDP,
    *,
    horizon: int = 24,
    gamma: float = 0.99,
) -> BellmanResult:
    """Resuelve la ecuación de Bellman por inducción hacia atrás en horizonte finito."""
    if horizon < 1:
        raise ValueError("horizon debe ser >= 1.")
    n_states = len(mdp.states)
    values = np.zeros((horizon + 1, n_states), dtype=float)
    q_values = np.full((horizon, n_states, len(ACCIONES)), np.nan, dtype=float)
    policy = np.full((horizon, n_states), "", dtype=object)

    for t in range(horizon - 1, -1, -1):
        for s_idx, state in enumerate(mdp.states):
            candidates = []
            for action in acciones_admisibles(state[2]):
                a_idx = ACCIONES.index(action)
                q_val = mdp.rewards[action][s_idx] + gamma * (
                    mdp.transitions[action][s_idx] @ values[t + 1]
                )
                q_values[t, s_idx, a_idx] = q_val
                candidates.append((q_val, action))
            best_val, best_action = max(candidates, key=lambda item: item[0])
            values[t, s_idx] = best_val
            policy[t, s_idx] = best_action

    return BellmanResult(values=values, q_values=q_values, policy=policy)


def tabla_politica(
    mdp: EmpiricalMDP,
    solution: BellmanResult,
    *,
    time: int = 0,
) -> pd.DataFrame:
    """Genera la tabla legible estado -> acción recomendada -> valor óptimo."""
    rows = []
    for s_idx, (regime, signal, pos) in enumerate(mdp.states):
        action = solution.policy[time, s_idx]
        rows.append({
            "regimen_mercado": regime,
            "senal_wallets": signal,
            "posicion": pos,
            "accion_recomendada": action,
            "valor_optimo": solution.values[time, s_idx],
        })
    return pd.DataFrame(rows)


def simular_politica(
    mdp: EmpiricalMDP,
    solution: BellmanResult,
    *,
    initial_state: State = ("FLAT", "NEUTRAL", 0),
    seed: int = 7,
    initial_wealth: float = 100.0,
) -> pd.DataFrame:
    """Simula una trayectoria estocástica del agente usando las transiciones aprendidas."""
    if initial_state not in mdp.states:
        raise ValueError("initial_state no pertenece al espacio de estados.")
    s_idx = mdp.states.index(initial_state)
    rng = np.random.default_rng(seed)
    wealth = float(initial_wealth)
    rows = []
    for t in range(len(solution.policy)):
        state = mdp.states[s_idx]
        action = str(solution.policy[t, s_idx])
        reward = float(mdp.rewards[action][s_idx])
        next_idx = int(rng.choice(len(mdp.states), p=mdp.transitions[action][s_idx]))
        wealth *= 1.0 + reward
        rows.append({
            "t": t, "estado": etiqueta_estado(state), "accion": action, "recompensa": reward,
            "siguiente_estado": etiqueta_estado(mdp.states[next_idx]), "wealth": wealth,
        })
        s_idx = next_idx
    return pd.DataFrame(rows)


def replay_historico(
    observaciones: pd.DataFrame,
    mdp: EmpiricalMDP,
    solution: BellmanResult,
    *,
    initial_position: int = 0,
    initial_wealth: float = 100.0,
) -> pd.DataFrame:
    """Aplica la política óptima sobre los datos históricos observados y registra riqueza."""
    state_index = {state: idx for idx, state in enumerate(mdp.states)}
    position = initial_position
    wealth = float(initial_wealth)
    rows = []
    for t, (_, row) in enumerate(observaciones.iloc[:len(solution.policy)].iterrows()):
        state = (row["regimen_mercado"], row["senal_wallets"], position)
        idx = state_index[state]
        action = str(solution.policy[t, idx])
        reward = calcular_recompensa_con_perfiles(
            posicion=position, accion=action, precio_t=float(row["precio_t"]),
            precio_t1=float(row["precio_t1"]), costo_gas_ratio=float(row["costo_gas_ratio"]),
            support_wallets=_support_from_row(row), profile_weight=mdp.profile_weight,
        )
        pos_after = posicion_despues(position, action)
        wealth *= 1.0 + reward["reward_final"]
        rows.append({
            "as_of": row.get("as_of"),
            "estado": etiqueta_estado(state),
            "accion": action,
            "posicion_despues": pos_after,
            "recompensa_base": reward["reward_base"],
            "bonus_wallets": reward["profile_bonus"],
            "recompensa_final": reward["reward_final"],
            "wealth": wealth,
        })
        position = pos_after
    return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────────────────────
# EJECUTOR INTEGRADO EN MEMORIA
# ──────────────────────────────────────────────────────────────────────────────

def ejecutar_agente_bellman(
    *,
    swaps_logicos: pd.DataFrame,
    ledger: pd.DataFrame,
    perfiles: pd.DataFrame,
    consistency_threshold: float = CONSISTENCY_THRESHOLD_RL,
    flat_band: float = 0.001,
    profile_weight: float = 0.25,
    laplace_alpha: float = 1.0,
    horizon: int = 24,
    gamma: float = 0.99,
) -> RLPipelineResult:
    """Ejecuta el pipeline completo de Bellman en memoria."""
    wallets_dirigidas = calcular_wallets_ganadoras_1h_consistentes(
        perfiles, consistency_threshold=consistency_threshold,
    )
    observaciones = construir_observaciones_horarias(
        swaps_logicos, ledger, consistency_threshold=consistency_threshold, flat_band=flat_band,
    )
    mdp = construir_mdp_empirico(
        observaciones, profile_weight=profile_weight, laplace_alpha=laplace_alpha,
    )
    bellman = resolver_bellman_finito(mdp, horizon=horizon, gamma=gamma)
    politica = tabla_politica(mdp, bellman, time=0)
    replay = replay_historico(observaciones, mdp, bellman)

    return RLPipelineResult(
        wallets_dirigidas=wallets_dirigidas,
        observaciones=observaciones,
        mdp=mdp,
        bellman=bellman,
        politica=politica,
        replay=replay,
        output_dir=None,
    )


__all__ = [
    "ACCIONES",
    "BellmanResult",
    "EmpiricalMDP",
    "REGIMENES_MERCADO",
    "RLPipelineResult",
    "SENALES_WALLETS",
    "acciones_admisibles",
    "calcular_recompensa_con_perfiles",
    "confianza_relativa_accion",
    "construir_mdp_empirico",
    "construir_observaciones_horarias",
    "desglose_recompensas",
    "ejecutar_agente_bellman",
    "estados_mdp",
    "etiqueta_estado",
    "posicion_despues",
    "replay_historico",
    "resolver_bellman_finito",
    "simular_politica",
    "tabla_politica",
    "verificar_probabilidades",
]
