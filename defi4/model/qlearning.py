"""Q-learning tabular para el mismo MDP causal de POL usado por Bellman.

Bellman conoce un modelo de transición y calcula hacia atrás. Este módulo no
consulta esa matriz durante el aprendizaje: recorre episodios históricos,
elige una acción, observa su recompensa y actualiza una sola entrada de Q.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .mdp import EmpiricalMDP, _support_from_row, etiqueta_estado
from .rewards import ACCIONES, acciones_admisibles, calcular_recompensa_con_perfiles, posicion_despues


@dataclass(frozen=True)
class QLearningResult:
    """Tabla Q, visitas y diagnóstico de entrenamiento por episodio."""

    q_values: np.ndarray
    visits: np.ndarray
    training_log: pd.DataFrame
    epsilon_start: float
    epsilon_end: float
    alpha: float
    gamma: float


def _epsilon_for_episode(episode: int, episodes: int, start: float, end: float) -> float:
    if episodes == 1:
        return float(end)
    progress = (episode - 1) / (episodes - 1)
    return float(start + progress * (end - start))


def _choose_action(
    q_row: np.ndarray,
    allowed: tuple[str, str],
    epsilon: float,
    random: np.random.Generator,
) -> tuple[str, bool]:
    """Elige ε-greedy, sorteando empates sólo entre acciones admisibles."""
    allowed_indices = np.array([ACCIONES.index(action) for action in allowed], dtype=int)
    explore = bool(random.random() < epsilon)
    if explore:
        action_index = int(random.choice(allowed_indices))
    else:
        allowed_values = q_row[allowed_indices]
        best = allowed_indices[np.isclose(allowed_values, allowed_values.max())]
        action_index = int(random.choice(best))
    return ACCIONES[action_index], explore


def entrenar_q_learning(
    observaciones: pd.DataFrame,
    mdp: EmpiricalMDP,
    *,
    episodes: int = 2_000,
    alpha: float = 0.15,
    gamma: float = 0.99,
    epsilon_start: float = 0.90,
    epsilon_end: float = 0.05,
    seed: int = 7,
) -> QLearningResult:
    """Aprende Q desde las experiencias observadas, sin usar ``P(s'|s,a)``.

    Cada episodio repite la trayectoria histórica disponible. En cada hora el
    estado usa sólo el régimen y la señal de wallets ya calculados para ese
    corte; la recompensa usa el precio de la hora siguiente, que se revela
    después de actuar. Es entrenamiento in-sample, no una prueba de retorno.
    """
    if episodes < 1:
        raise ValueError("episodes debe ser >= 1.")
    if not 0.0 < alpha <= 1.0:
        raise ValueError("alpha debe estar en (0, 1].")
    if not 0.0 <= gamma <= 1.0:
        raise ValueError("gamma debe estar entre 0 y 1.")
    if not 0.0 <= epsilon_end <= epsilon_start <= 1.0:
        raise ValueError("epsilon debe cumplir 0 <= final <= inicial <= 1.")
    if observaciones.empty:
        raise ValueError("Se requiere al menos una observación para entrenar Q-learning.")

    frame = observaciones.reset_index(drop=True)
    state_index = {state: index for index, state in enumerate(mdp.states)}
    horizon = len(frame)
    q_values = np.zeros((horizon, len(mdp.states), len(ACCIONES)), dtype=float)
    visits = np.zeros_like(q_values, dtype=int)
    random = np.random.default_rng(seed)
    log_rows = []

    for episode in range(1, episodes + 1):
        epsilon = _epsilon_for_episode(episode, episodes, epsilon_start, epsilon_end)
        position = 0
        total_reward = 0.0
        for time, (_, row) in enumerate(frame.iterrows()):
            state = (row["regimen_mercado"], row["senal_wallets"], position)
            current_index = state_index[state]
            action, explore = _choose_action(
                q_values[time, current_index], acciones_admisibles(position), epsilon, random,
            )
            action_index = ACCIONES.index(action)
            reward = calcular_recompensa_con_perfiles(
                posicion=position, accion=action, precio_t=float(row["precio_t"]),
                precio_t1=float(row["precio_t1"]), costo_gas_ratio=float(row["costo_gas_ratio"]),
                support_wallets=_support_from_row(row), profile_weight=mdp.profile_weight,
            )
            next_position = posicion_despues(position, action)
            terminal = time == horizon - 1
            if terminal:
                next_max = 0.0
            else:
                next_row = frame.iloc[time + 1]
                next_state = (next_row["regimen_mercado"], next_row["senal_wallets"], next_position)
                next_index = state_index[next_state]
                next_allowed = [ACCIONES.index(candidate) for candidate in acciones_admisibles(next_position)]
                next_max = float(q_values[time + 1, next_index, next_allowed].max())
            old_value = float(q_values[time, current_index, action_index])
            target = float(reward["reward_final"]) + gamma * next_max
            q_values[time, current_index, action_index] = old_value + alpha * (target - old_value)
            visits[time, current_index, action_index] += 1
            total_reward += float(reward["reward_final"])
            position = next_position
        log_rows.append({
            "episode": episode, "epsilon": epsilon, "recompensa_total": total_reward,
            "q_abs_media": float(np.abs(q_values).mean()),
        })

    return QLearningResult(
        q_values=q_values, visits=visits, training_log=pd.DataFrame(log_rows),
        epsilon_start=epsilon_start, epsilon_end=epsilon_end, alpha=alpha, gamma=gamma,
    )


def tabla_politica_q_learning(
    mdp: EmpiricalMDP,
    solution: QLearningResult,
    *,
    time: int = 0,
) -> pd.DataFrame:
    """Expone la política greedy y los valores Q de cada estado al tiempo dado."""
    if not 0 <= time < len(solution.q_values):
        raise ValueError("time debe existir dentro del horizonte entrenado.")
    rows = []
    for state_index, (regime, signal, position) in enumerate(mdp.states):
        allowed = acciones_admisibles(position)
        allowed_indices = [ACCIONES.index(action) for action in allowed]
        values = solution.q_values[time, state_index, allowed_indices]
        action = allowed[int(np.argmax(values))]
        row = {
            "regimen_mercado": regime, "senal_wallets": signal, "posicion": position,
            "accion_recomendada": action,
            "valor_q": float(solution.q_values[time, state_index, ACCIONES.index(action)]),
        }
        for candidate in ACCIONES:
            index = ACCIONES.index(candidate)
            row[f"q_{candidate.lower()}"] = (
                float(solution.q_values[time, state_index, index]) if candidate in allowed else np.nan
            )
            row[f"visitas_{candidate.lower()}"] = (
                int(solution.visits[time, state_index, index]) if candidate in allowed else 0
            )
        rows.append(row)
    return pd.DataFrame(rows)


def replay_q_learning_historico(
    observaciones: pd.DataFrame,
    mdp: EmpiricalMDP,
    solution: QLearningResult,
    *,
    initial_position: int = 0,
    initial_wealth: float = 100.0,
) -> pd.DataFrame:
    """Evalúa la política greedy congelada en la trayectoria histórica."""
    if initial_position not in (0, 1):
        raise ValueError("initial_position debe ser 0 o 1.")
    frame = observaciones.iloc[:len(solution.q_values)].reset_index(drop=True)
    state_index = {state: index for index, state in enumerate(mdp.states)}
    position = initial_position
    wealth_objective = float(initial_wealth)
    wealth_economic = float(initial_wealth)
    rows = []
    for time, (_, row) in enumerate(frame.iterrows()):
        state = (row["regimen_mercado"], row["senal_wallets"], position)
        current_index = state_index[state]
        allowed = acciones_admisibles(position)
        allowed_indices = [ACCIONES.index(action) for action in allowed]
        values = solution.q_values[time, current_index, allowed_indices]
        action = allowed[int(np.argmax(values))]
        reward = calcular_recompensa_con_perfiles(
            posicion=position, accion=action, precio_t=float(row["precio_t"]),
            precio_t1=float(row["precio_t1"]), costo_gas_ratio=float(row["costo_gas_ratio"]),
            support_wallets=_support_from_row(row), profile_weight=mdp.profile_weight,
        )
        wealth_objective *= 1.0 + float(reward["reward_final"])
        wealth_economic *= 1.0 + float(reward["reward_base"])
        position_after = posicion_despues(position, action)
        rows.append({
            "as_of": row.get("as_of"), "estado": etiqueta_estado(state), "accion": action,
            "posicion_despues": position_after, "recompensa_base": reward["reward_base"],
            "bonus_wallets": reward["profile_bonus"], "recompensa_final": reward["reward_final"],
            "wealth": wealth_objective, "wealth_economica": wealth_economic,
        })
        position = position_after
    return pd.DataFrame(rows)


def replay_buy_and_hold(
    observaciones: pd.DataFrame,
    *,
    initial_wealth: float = 100.0,
) -> pd.DataFrame:
    """Línea base: comprar POL en la primera hora y mantener hasta el final."""
    if observaciones.empty:
        raise ValueError("Se requiere al menos una observación para Buy and Hold.")
    wealth = float(initial_wealth)
    rows = []
    for time, (_, row) in enumerate(observaciones.reset_index(drop=True).iterrows()):
        action = "BUY_POL" if time == 0 else "HOLD"
        gas = float(row["costo_gas_ratio"]) if time == 0 else 0.0
        gross_return = float(row["precio_t1"]) / float(row["precio_t"]) - 1.0
        reward = gross_return - gas
        wealth *= 1.0 + reward
        rows.append({
            "as_of": row.get("as_of"), "accion": action, "recompensa_base": reward,
            "wealth_economica": wealth,
        })
    return pd.DataFrame(rows)
