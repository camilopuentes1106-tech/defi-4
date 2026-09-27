import unittest

import pandas as pd

from defi4.model import (
    construir_mdp_empirico,
    entrenar_q_learning,
    replay_buy_and_hold,
    replay_q_learning_historico,
    tabla_politica_q_learning,
)
from defi4.model.rewards import acciones_admisibles


def _observaciones() -> pd.DataFrame:
    return pd.DataFrame([
        {"as_of": "2026-08-22T00:00:00Z", "regimen_mercado": "UP", "senal_wallets": "BUY", "precio_t": 100.0, "precio_t1": 101.0, "costo_gas_ratio": 0.001, "support_buy": 0.9, "support_sell": 0.1, "support_hold": 0.0},
        {"as_of": "2026-08-22T01:00:00Z", "regimen_mercado": "FLAT", "senal_wallets": "NEUTRAL", "precio_t": 101.0, "precio_t1": 100.5, "costo_gas_ratio": 0.001, "support_buy": 0.0, "support_sell": 0.0, "support_hold": 0.0},
        {"as_of": "2026-08-22T02:00:00Z", "regimen_mercado": "DOWN", "senal_wallets": "SELL", "precio_t": 100.5, "precio_t1": 99.0, "costo_gas_ratio": 0.001, "support_buy": 0.1, "support_sell": 0.9, "support_hold": 0.0},
        {"as_of": "2026-08-22T03:00:00Z", "regimen_mercado": "UP", "senal_wallets": "HOLD", "precio_t": 99.0, "precio_t1": 100.0, "costo_gas_ratio": 0.001, "support_buy": 0.1, "support_sell": 0.1, "support_hold": 0.9},
    ])


class QLearningTests(unittest.TestCase):
    def test_q_learning_replay_and_buy_hold_are_reproducible(self):
        observations = _observaciones()
        mdp = construir_mdp_empirico(observations)
        solution = entrenar_q_learning(
            observations, mdp, episodes=80, alpha=0.2, epsilon_start=0.4, epsilon_end=0.05, seed=3,
        )
        policy = tabla_politica_q_learning(mdp, solution)
        replay = replay_q_learning_historico(observations, mdp, solution)
        buy_hold = replay_buy_and_hold(observations)

        self.assertEqual(solution.q_values.shape, (4, 24, 3))
        self.assertEqual(int(solution.visits.sum()), 80 * 4)
        self.assertEqual(len(replay), 4)
        self.assertEqual(len(buy_hold), 4)
        self.assertTrue((replay["wealth_economica"] > 0).all())
        self.assertTrue((buy_hold["wealth_economica"] > 0).all())
        for _, row in policy.iterrows():
            self.assertIn(row["accion_recomendada"], acciones_admisibles(int(row["posicion"])))


if __name__ == "__main__":
    unittest.main()
