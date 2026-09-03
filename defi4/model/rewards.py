"""Recompensas financieras con refuerzo prudente de perfiles de wallets.

La rentabilidad neta del portafolio es siempre el objetivo principal. Los
``consistency_score`` de las wallets sólo amplifican una acción cuando la
evidencia dirigida y el resultado financiero coinciden; nunca otorgan una
recompensa fija sólo por seguir una wallet.
"""

from __future__ import annotations

from typing import Mapping

import numpy as np


ACCIONES = ("BUY_POL", "SELL_POL", "HOLD")
_SUPPORT_KEYS = {
    "BUY_POL": "support_buy",
    "SELL_POL": "support_sell",
    "HOLD": "support_hold",
}


def acciones_admisibles(posicion: int) -> tuple[str, str]:
    """Acciones válidas para un portafolio long-only binario."""
    if posicion == 0:
        return "BUY_POL", "HOLD"
    if posicion == 1:
        return "SELL_POL", "HOLD"
    raise ValueError("posicion debe ser 0 (USDC) o 1 (POL).")


def posicion_despues(posicion: int, accion: str) -> int:
    """Aplica la acción al inventario, validando que sea admisible."""
    if accion not in acciones_admisibles(posicion):
        raise ValueError(f"{accion} no es admisible para posicion={posicion}.")
    if accion == "BUY_POL":
        return 1
    if accion == "SELL_POL":
        return 0
    return posicion


def _validar_entrada(
    precio_t: float,
    precio_t1: float,
    costo_gas_ratio: float,
    profile_weight: float,
) -> None:
    if not np.isfinite(precio_t) or precio_t <= 0:
        raise ValueError("precio_t debe ser positivo.")
    if not np.isfinite(precio_t1) or precio_t1 <= 0:
        raise ValueError("precio_t1 debe ser positivo.")
    if not np.isfinite(costo_gas_ratio) or costo_gas_ratio < 0:
        raise ValueError("costo_gas_ratio debe ser >= 0.")
    if not 0.0 <= profile_weight <= 1.0:
        raise ValueError("profile_weight debe estar entre 0 y 1.")


def _support(accion: str, support_wallets: Mapping[str, float | int]) -> float:
    key = _SUPPORT_KEYS[accion]
    value = float(support_wallets.get(key, 0.0))
    return max(value, 0.0) if np.isfinite(value) else 0.0


def confianza_relativa_accion(
    accion: str,
    support_wallets: Mapping[str, float | int],
) -> float:
    """Confianza 0..1 sólo si la acción domina a las otras direcciones."""
    if accion not in ACCIONES:
        raise ValueError(f"Acción no soportada: {accion}.")
    supports = {candidate: _support(candidate, support_wallets) for candidate in ACCIONES}
    leader = max(supports, key=supports.get)
    if supports[leader] <= 0 or accion != leader:
        return 0.0
    if sum(np.isclose(value, supports[leader]) for value in supports.values()) > 1:
        return 0.0
    next_best = max(value for candidate, value in supports.items() if candidate != accion)
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
    """Calcula base, ventaja y refuerzo de perfiles para las acciones válidas.

    ``reward_final = reward_base + λ × confianza × ventaja``.
    La ventaja compara una acción con la otra alternativa admisible; por eso
    la señal de wallets sólo suma valor cuando coincide con una mejor decisión
    financiera realizada.
    """
    _validar_entrada(precio_t, precio_t1, costo_gas_ratio, profile_weight)
    support_wallets = support_wallets or {}
    return_pol = precio_t1 / precio_t - 1.0
    base: dict[str, float] = {}
    for accion in acciones_admisibles(posicion):
        next_position = posicion_despues(posicion, accion)
        cost = costo_gas_ratio if accion in {"BUY_POL", "SELL_POL"} else 0.0
        base[accion] = next_position * return_pol - cost

    result: dict[str, dict[str, float]] = {}
    for accion, reward_base in base.items():
        alternatives = [value for candidate, value in base.items() if candidate != accion]
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
