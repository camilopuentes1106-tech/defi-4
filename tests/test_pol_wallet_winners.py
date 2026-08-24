import unittest

import pandas as pd

from pol_wallet_winners import (
    calcular_consistency_score,
    clasificar_winner_status,
    filtrar_wallets_ganadoras,
)


class WinnerRulesTests(unittest.TestCase):
    def test_score_uses_the_defined_weights_and_caps_profit_factor(self):
        score = calcular_consistency_score(0.75, 7.0, 4)
        self.assertAlmostEqual(score, 0.755)

    def test_winner_requires_all_four_conditions(self):
        status = clasificar_winner_status(
            n_decisiones=3,
            n_pendientes=0,
            pnl_neto_usdc=2.0,
            retorno_neto_mediano=0.01,
            consistency_score=0.66,
        )
        self.assertEqual(status, "winner")
        self.assertEqual(clasificar_winner_status(
            n_decisiones=2,
            n_pendientes=0,
            pnl_neto_usdc=2.0,
            retorno_neto_mediano=0.01,
            consistency_score=0.90,
        ), "insufficient")

    def test_filter_keeps_the_winning_action_horizon_profiles(self):
        profiles = pd.DataFrame([
            {"wallet": "0xa", "accion": "BUY_POL", "horizonte": "1h", "winner_status": "winner", "consistency_score": 0.7},
            {"wallet": "0xa", "accion": "SELL_POL", "horizonte": "1h", "winner_status": "not_winner", "consistency_score": 0.9},
            {"wallet": "0xb", "accion": "HOLD", "horizonte": "4h", "winner_status": "winner", "consistency_score": 0.8},
        ])
        winners = filtrar_wallets_ganadoras(profiles, acciones=["BUY_POL", "HOLD"])

        self.assertEqual(winners["wallet"].tolist(), ["0xb", "0xa"])
        self.assertEqual(winners["accion"].tolist(), ["HOLD", "BUY_POL"])


if __name__ == "__main__":
    unittest.main()
