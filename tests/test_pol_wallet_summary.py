import unittest

import numpy as np
import pandas as pd

from pol_wallet_summary import (
    construir_perfiles_desde_tabla_wallets,
    filtrar_candidatas_wallets,
)


class WalletSummaryTests(unittest.TestCase):
    def setUp(self):
        self.wallets = pd.DataFrame([
            {
                "wallet": "0xAAA", "direccion": "Buy", "n_swaps": 3,
                "pol_cantidad": 10.0, "usdc_cantidad": 20.0, "gas_usdc": 1.0,
                "usdc_neto": 19.0, "ganancia_neta_usdc_1m": 2.0,
                "ganancia_neta_usdc_5m": -1.0, "ganancia_neta_usdc_15m": 3.0,
                "ganancia_neta_usdc_1h": 4.0,
            },
            {
                "wallet": "0xBBB", "direccion": "Sell", "n_swaps": 2,
                "pol_cantidad": 8.0, "usdc_cantidad": 16.0, "gas_usdc": 0.5,
                "usdc_neto": 15.5, "ganancia_neta_usdc_1m": 5.0,
                "ganancia_neta_usdc_5m": np.nan, "ganancia_neta_usdc_15m": 1.0,
                "ganancia_neta_usdc_1h": 2.0,
            },
        ])

    def test_table_is_ranked_by_wallet_position_and_requested_horizon(self):
        profiles = construir_perfiles_desde_tabla_wallets(self.wallets)

        buy_1m = profiles[(profiles["wallet"] == "0xaaa") & (profiles["horizonte"] == "1m")].iloc[0]
        sell_1m = profiles[(profiles["wallet"] == "0xbbb") & (profiles["horizonte"] == "1m")].iloc[0]
        sell_5m = profiles[(profiles["wallet"] == "0xbbb") & (profiles["horizonte"] == "5m")].iloc[0]

        self.assertEqual(buy_1m["accion"], "BUY_POL")
        self.assertEqual(buy_1m["estado_resumen"], "candidate_winner")
        self.assertAlmostEqual(buy_1m["retorno_neto_agregado"], 2.0 / 19.0)
        self.assertEqual(sell_1m["estado_resumen"], "insufficient")
        self.assertEqual(sell_5m["estado_resumen"], "missing_return")

    def test_candidates_do_not_overstate_the_final_winner_label(self):
        profiles = construir_perfiles_desde_tabla_wallets(self.wallets)
        candidates = filtrar_candidatas_wallets(profiles, acciones=["BUY_POL"], horizontes=["1h"])

        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates.iloc[0]["estado_resumen"], "candidate_winner")
        self.assertNotIn("winner_status", profiles.columns)


if __name__ == "__main__":
    unittest.main()
