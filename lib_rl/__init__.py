"""Librería lib_rl: trading POL guiado por wallets on-chain y Bellman RL.

Módulos principales:
- transacciones: Ingesta on-chain (Alchemy RPC), gas POL/USDC y retornos forward.
- wallets: Agregación, ciclos FIFO, ledger maduro, perfiles y clasificación de ganadoras.
- bellman: MDP empírico 24 estados, reward shaping con ventaja neta y solución Bellman finito.
"""

from __future__ import annotations

from .transacciones import (
    DEFAULT_POOL_ADDRESS,
    DEFAULT_TICKER_YF,
    INTERVALOS_YF,
    SwapPipeline,
)
from .wallets import (
    CONSISTENCY_THRESHOLD_RL,
    MIN_CONSISTENCY_SCORE,
    MIN_DECISIONES_GANADORA,
    WalletAnalysisResult,
    WalletView,
    analizar_wallets,
    calcular_consenso_wallets_1h,
    calcular_consistency_score,
    calcular_wallets_ganadoras_1h_consistentes,
    canonicalizar_swaps_logicos,
    clasificar_winner_status,
    construir_ciclos_hold,
    construir_ledger_decisiones,
    construir_perfiles_causales_1h,
    construir_perfiles_wallet,
    construir_precios_por_minuto_desde_swaps,
    construir_senales_wallets_horarias,
    filtrar_wallets_ganadoras,
    resumen_estados_perfiles,
    seleccionar_wallets_dirigidas_1h,
)
from .bellman import (
    ACCIONES,
    REGIMENES_MERCADO,
    SENALES_WALLETS,
    BellmanResult,
    EmpiricalMDP,
    RLPipelineResult,
    acciones_admisibles,
    calcular_recompensa_con_perfiles,
    confianza_relativa_accion,
    construir_mdp_empirico,
    construir_observaciones_horarias,
    desglose_recompensas,
    ejecutar_agente_bellman,
    estados_mdp,
    etiqueta_estado,
    posicion_despues,
    replay_historico,
    resolver_bellman_finito,
    simular_politica,
    tabla_politica,
    verificar_probabilidades,
)

__all__ = [
    # Transacciones
    "DEFAULT_POOL_ADDRESS",
    "DEFAULT_TICKER_YF",
    "INTERVALOS_YF",
    "SwapPipeline",
    # Wallets
    "CONSISTENCY_THRESHOLD_RL",
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
    # Bellman
    "ACCIONES",
    "REGIMENES_MERCADO",
    "SENALES_WALLETS",
    "BellmanResult",
    "EmpiricalMDP",
    "RLPipelineResult",
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
