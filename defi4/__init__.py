"""Librería DEFI IV para perfiles on-chain de POL y el agente Bellman.

La API pública se organiza por responsabilidad:

* :mod:`defi4.data`: descarga y normalización de swaps POL/USDC.
* :mod:`defi4.wallets`: perfiles, filtros de wallets ganadoras e informes.
* :mod:`defi4.model`: estado, recompensa, transiciones y política Bellman.
* :mod:`defi4.pipeline`: flujos reproducibles de punta a punta.
"""

from . import data, model, pipeline, wallets

__all__ = ["data", "model", "pipeline", "wallets"]
