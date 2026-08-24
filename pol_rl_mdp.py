"""MDP empírico y Bellman finito para explicar el agente POL paso a paso.

El agente opera cada hora. Su estado combina un régimen de mercado calculado
con retornos 1m/5m/15m/1h, una señal causal de wallets ganadoras 1h y su
posición binaria (USDC o POL). El módulo estima un modelo de transición desde
la trayectoria observada y resuelve Bellman por programación dinámica.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Iterable, Mapping

import numpy as np
import pandas as pd

from pol_rl_rewards import (
    ACCIONES,
    acciones_admisibles,
    calcular_recompensa_con_perfiles,
    posicion_despues,
)
from pol_rl_wallet_signal import (
    CONSISTENCY_THRESHOLD_RL,
    construir_senales_wallets_horarias,
)


REGIMENES_MERCADO = ("DOWN", "FLAT", "UP")
SENALES_WALLETS = ("BUY", "SELL", "HOLD", "NEUTRAL")

State = tuple[str, str, int]


def etiqueta_estado(state: State) -> str:
    """Representación legible y exportable de un estado del MDP."""
    regime, signal, position = state
    return f"{regime} | {signal} | posición={position}"


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
    """Valores, Q-valores y política de un horizonte finito."""

    values: np.ndarray
    q_values: np.ndarray
    policy: np.ndarray


def _as_utc_series(values: Iterable[object]) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(pd.to_datetime(list(values), utc=True, errors="coerce"))


def _last_price_before(prices: pd.Series, timestamps: pd.DatetimeIndex) -> pd.Series:
    """Último precio conocido al corte de cada timestamp, sin mirar el futuro."""
    source = (
        prices.dropna()
        .sort_index()
        .rename("price")
        .rename_axis("price_time")
        .reset_index()
    )
    target = pd.DataFrame({"timestamp": timestamps}).sort_values("timestamp")
    merged = pd.merge_asof(target, source, left_on="timestamp", right_on="price_time", direction="backward")
    return merged.set_index("timestamp")["price"].reindex(timestamps)


def _price_events(swaps_logicos: pd.DataFrame) -> pd.Series:
    required = {"timestamp", "precio_ejecutado", "direccion"}
    missing = required.difference(swaps_logicos.columns)
    if missing:
        raise ValueError("Faltan columnas de swaps lógicos: " + ", ".join(sorted(missing)))
    frame = swaps_logicos.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    frame["precio_ejecutado"] = pd.to_numeric(frame["precio_ejecutado"], errors="coerce")
    frame = frame[
        frame["timestamp"].notna()
        & (frame["precio_ejecutado"] > 0)
        & frame["direccion"].isin(["BUY_POL", "SELL_POL"])
    ].sort_values(["timestamp", "hash_tx"] if "hash_tx" in frame.columns else ["timestamp"])
    if frame.empty:
        return pd.Series(dtype=float, name="price")
    prices = frame.set_index("timestamp")["precio_ejecutado"]
    return prices[~prices.index.duplicated(keep="last")].rename("price")


def _gas_cost_ratio(swaps_logicos: pd.DataFrame) -> float:
    if not {"gas_usdc", "notional_usdc"}.issubset(swaps_logicos.columns):
        return 0.0
    notional = pd.to_numeric(swaps_logicos["notional_usdc"], errors="coerce")
    gas = pd.to_numeric(swaps_logicos["gas_usdc"], errors="coerce")
    ratios = (gas / notional).replace([np.inf, -np.inf], np.nan).dropna()
    return float(ratios.median()) if not ratios.empty else 0.0


def _clasificar_regimen(momentum: pd.Series, *, flat_band: float) -> tuple[pd.Series, float, float]:
    """Clasifica el impulso con una banda fija, no con datos futuros."""
    if flat_band < 0:
        raise ValueError("flat_band debe ser >= 0.")
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
    """Construye la tabla causal que alimenta estados y recompensas del MDP.

    Cada fila representa una decisión en el corte de una hora y contiene el
    precio de esa hora, los retornos pasados de los cuatro horizontes y la
    señal de wallets disponible en ese momento. ``flat_band`` es la variación
    absoluta que se considera mercado lateral (por defecto 0.10 %); es fija
    para no introducir información futura. La última hora sin precio
    posterior se descarta porque no puede producir una recompensa observada.
    """
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
    frame["regimen_mercado"], lower, upper = _clasificar_regimen(
        frame["momentum_compuesto"], flat_band=flat_band,
    )
    signals = construir_senales_wallets_horarias(
        ledger, frame["as_of"], consistency_threshold=consistency_threshold,
    )
    frame = frame.merge(signals, on="as_of", how="left")
    frame["senal_wallets"] = frame["senal_wallets"].fillna("NEUTRAL")
    for column in ("confianza_wallets", "support_buy", "support_sell", "support_hold"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0.0)
    frame["costo_gas_ratio"] = _gas_cost_ratio(swaps_logicos)
    frame.attrs["momentum_lower_threshold"] = lower
    frame.attrs["momentum_upper_threshold"] = upper
    return frame.reset_index(drop=True)


def estados_mdp() -> tuple[State, ...]:
    """Espacio de 24 estados interpretables: régimen × señal × posición."""
    return tuple((regime, signal, position) for regime, signal, position in product(
        REGIMENES_MERCADO, SENALES_WALLETS, (0, 1)
    ))


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
    """Estima ``P(s'|s,a)`` y recompensa media para el MDP de Bellman."""
    if laplace_alpha <= 0:
        raise ValueError("laplace_alpha debe ser > 0.")
    required = {
        "regimen_mercado", "senal_wallets", "precio_t", "precio_t1",
        "costo_gas_ratio", "support_buy", "support_sell", "support_hold",
    }
    missing = required.difference(observaciones.columns)
    if missing:
        raise ValueError("Faltan columnas de observaciones: " + ", ".join(sorted(missing)))
    frame = observaciones.copy().reset_index(drop=True)
    frame = frame[
        frame["regimen_mercado"].isin(REGIMENES_MERCADO)
        & frame["senal_wallets"].isin(SENALES_WALLETS)
    ].copy()
    if len(frame) < 2:
        raise ValueError("Se requieren al menos dos observaciones horarias para estimar transiciones.")

    economic_states = tuple(product(REGIMENES_MERCADO, SENALES_WALLETS))
    economic_index = {state: index for index, state in enumerate(economic_states)}
    states = estados_mdp()
    state_index = {state: index for index, state in enumerate(states)}
    counts = np.full((len(economic_states), len(economic_states)), laplace_alpha, dtype=float)
    for index in range(len(frame) - 1):
        source = (frame.loc[index, "regimen_mercado"], frame.loc[index, "senal_wallets"])
        target = (frame.loc[index + 1, "regimen_mercado"], frame.loc[index + 1, "senal_wallets"])
        counts[economic_index[source], economic_index[target]] += 1.0
    economic_transition = counts / counts.sum(axis=1, keepdims=True)

    transitions = {action: np.zeros((len(states), len(states)), dtype=float) for action in ACCIONES}
    for source_index, (regime, signal, position) in enumerate(states):
        source_economic = economic_index[(regime, signal)]
        for action in acciones_admisibles(position):
            next_position = posicion_despues(position, action)
            for target_economic, probability in zip(economic_states, economic_transition[source_economic]):
                target_index = state_index[(*target_economic, next_position)]
                transitions[action][source_index, target_index] = probability

    rewards = {action: np.full(len(states), np.nan, dtype=float) for action in ACCIONES}
    reward_samples: dict[tuple[tuple[str, str], int, str], list[float]] = {}
    global_samples: dict[tuple[int, str], list[float]] = {}
    for _, row in frame.iterrows():
        economic = (row["regimen_mercado"], row["senal_wallets"])
        for position in (0, 1):
            for action in acciones_admisibles(position):
                reward = calcular_recompensa_con_perfiles(
                    posicion=position, accion=action, precio_t=float(row["precio_t"]),
                    precio_t1=float(row["precio_t1"]), costo_gas_ratio=float(row["costo_gas_ratio"]),
                    support_wallets=_support_from_row(row), profile_weight=profile_weight,
                )["reward_final"]
                reward_samples.setdefault((economic, position, action), []).append(reward)
                global_samples.setdefault((position, action), []).append(reward)
    for state_i, (regime, signal, position) in enumerate(states):
        economic = (regime, signal)
        for action in acciones_admisibles(position):
            samples = reward_samples.get((economic, position, action), global_samples[(position, action)])
            rewards[action][state_i] = float(np.mean(samples))

    return EmpiricalMDP(
        states=states, transitions=transitions, rewards=rewards,
        gas_cost_ratio=float(frame["costo_gas_ratio"].median()), profile_weight=float(profile_weight),
    )


def verificar_probabilidades(mdp: EmpiricalMDP, *, atol: float = 1e-10) -> pd.DataFrame:
    """Devuelve una fila por estado/acción válida y verifica que la fila de P suma 1."""
    rows = []
    for state_index, state in enumerate(mdp.states):
        for action in acciones_admisibles(state[2]):
            total = float(mdp.transitions[action][state_index].sum())
            rows.append({
                "estado": etiqueta_estado(state), "accion": action, "suma_probabilidades": total,
                "valida": bool(np.isclose(total, 1.0, atol=atol)),
            })
    return pd.DataFrame(rows)


def resolver_bellman_finito(
    mdp: EmpiricalMDP,
    *,
    horizon: int = 24,
    gamma: float = 0.99,
) -> BellmanResult:
    """Resuelve Bellman hacia atrás y devuelve ``V``, ``Q`` y ``π*``."""
    if horizon < 1:
        raise ValueError("horizon debe ser >= 1.")
    if not 0.0 <= gamma <= 1.0:
        raise ValueError("gamma debe estar entre 0 y 1.")
    n_states = len(mdp.states)
    values = np.zeros((horizon + 1, n_states), dtype=float)
    q_values = np.full((horizon, n_states, len(ACCIONES)), np.nan, dtype=float)
    policy = np.full((horizon, n_states), "", dtype=object)
    for time in range(horizon - 1, -1, -1):
        for state_index, state in enumerate(mdp.states):
            candidates = []
            for action in acciones_admisibles(state[2]):
                action_index = ACCIONES.index(action)
                q_value = mdp.rewards[action][state_index] + gamma * (
                    mdp.transitions[action][state_index] @ values[time + 1]
                )
                q_values[time, state_index, action_index] = q_value
                candidates.append((q_value, action))
            best_value, best_action = max(candidates, key=lambda item: item[0])
            values[time, state_index] = best_value
            policy[time, state_index] = best_action
    return BellmanResult(values=values, q_values=q_values, policy=policy)


def tabla_politica(
    mdp: EmpiricalMDP,
    solution: BellmanResult,
    *,
    time: int = 0,
) -> pd.DataFrame:
    """Tabla didáctica ``estado → acción → valor`` para el notebook."""
    if not 0 <= time < len(solution.policy):
        raise ValueError("time debe existir dentro del horizonte resuelto.")
    rows = []
    for state_index, (regime, signal, position) in enumerate(mdp.states):
        action = solution.policy[time, state_index]
        rows.append({
            "regimen_mercado": regime, "senal_wallets": signal, "posicion": position,
            "accion_recomendada": action, "valor_optimo": solution.values[time, state_index],
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
    """Simula una trayectoria futura muestreando el modelo empírico estimado."""
    if initial_state not in mdp.states:
        raise ValueError("initial_state no pertenece al espacio de estados.")
    state_index = mdp.states.index(initial_state)
    random = np.random.default_rng(seed)
    wealth = float(initial_wealth)
    rows = []
    for time in range(len(solution.policy)):
        state = mdp.states[state_index]
        action = str(solution.policy[time, state_index])
        reward = float(mdp.rewards[action][state_index])
        next_index = int(random.choice(len(mdp.states), p=mdp.transitions[action][state_index]))
        wealth *= 1.0 + reward
        rows.append({
            "t": time, "estado": etiqueta_estado(state), "accion": action, "recompensa": reward,
            "siguiente_estado": etiqueta_estado(mdp.states[next_index]), "wealth": wealth,
        })
        state_index = next_index
    return pd.DataFrame(rows)


def replay_historico(
    observaciones: pd.DataFrame,
    mdp: EmpiricalMDP,
    solution: BellmanResult,
    *,
    initial_position: int = 0,
    initial_wealth: float = 100.0,
) -> pd.DataFrame:
    """Aplica ``π*`` sobre los precios observados para explicarlo hora a hora."""
    if initial_position not in (0, 1):
        raise ValueError("initial_position debe ser 0 o 1.")
    state_index = {state: index for index, state in enumerate(mdp.states)}
    position = initial_position
    wealth = float(initial_wealth)
    rows = []
    for time, (_, row) in enumerate(observaciones.iloc[:len(solution.policy)].iterrows()):
        state = (row["regimen_mercado"], row["senal_wallets"], position)
        index = state_index[state]
        action = str(solution.policy[time, index])
        reward = calcular_recompensa_con_perfiles(
            posicion=position, accion=action, precio_t=float(row["precio_t"]),
            precio_t1=float(row["precio_t1"]), costo_gas_ratio=float(row["costo_gas_ratio"]),
            support_wallets=_support_from_row(row), profile_weight=mdp.profile_weight,
        )
        position_after = posicion_despues(position, action)
        wealth *= 1.0 + reward["reward_final"]
        rows.append({
            "as_of": row.get("as_of"), "estado": etiqueta_estado(state), "accion": action,
            "posicion_despues": position_after, "recompensa_base": reward["reward_base"],
            "bonus_wallets": reward["profile_bonus"], "recompensa_final": reward["reward_final"],
            "wealth": wealth,
        })
        position = position_after
    return pd.DataFrame(rows)
