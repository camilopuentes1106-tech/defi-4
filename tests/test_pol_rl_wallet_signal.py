import unittest

import pandas as pd

from defi4.wallets.signals import (
    calcular_consenso_wallets_1h,
    construir_perfiles_causales_1h,
    seleccionar_wallets_dirigidas_1h,
)


T0 = pd.Timestamp("2026-08-22T00:00:00Z")


def decision(wallet, action, hour, net_return=0.10, pnl=1.0):
    return {
        "wallet": wallet,
        "accion": action,
        "horizonte": "1h",
        "decision_time": T0 + pd.Timedelta(hours=hour),
        "forward_time": T0 + pd.Timedelta(hours=hour + 1),
        "retorno_neto": net_return,
        "ganancia_neta_usdc": pnl,
    }


class WalletSignalTests(unittest.TestCase):
    def test_static_selection_requires_winner_1h_and_score_at_least_point_8(self):
        profiles = pd.DataFrame([
            {"wallet": "0xA", "accion": "BUY_POL", "horizonte": "1h", "winner_status": "winner", "consistency_score": 0.80},
            {"wallet": "0xB", "accion": "SELL_POL", "horizonte": "1h", "winner_status": "winner", "consistency_score": 0.79},
            {"wallet": "0xC", "accion": "HOLD", "horizonte": "15m", "winner_status": "winner", "consistency_score": 0.95},
            {"wallet": "0xD", "accion": "HOLD", "horizonte": "1h", "winner_status": "not_winner", "consistency_score": 0.99},
        ])
        selected = seleccionar_wallets_dirigidas_1h(profiles)

        self.assertEqual(selected["wallet"].tolist(), ["0xa"])
        self.assertEqual(selected.iloc[0]["direccion_agente"], "BUY")

    def test_causal_profiles_do_not_use_decisions_that_mature_after_cutoff(self):
        ledger = pd.DataFrame([
            decision("0xA", "BUY_POL", 0),
            decision("0xA", "BUY_POL", 1),
            decision("0xA", "BUY_POL", 2),
            decision("0xB", "SELL_POL", 3),
        ])

        early = construir_perfiles_causales_1h(ledger, as_of=T0 + pd.Timedelta(hours=2))
        mature = construir_perfiles_causales_1h(ledger, as_of=T0 + pd.Timedelta(hours=3))
        consensus = calcular_consenso_wallets_1h(mature)
        early_buy = early[early["accion"] == "BUY_POL"].iloc[0]
        mature_buy = mature[mature["accion"] == "BUY_POL"].iloc[0]

        self.assertEqual(early_buy["winner_status"], "insufficient")
        self.assertEqual(mature_buy["n_decisiones"], 3)
        self.assertEqual(mature_buy["winner_status"], "winner")
        self.assertGreaterEqual(mature_buy["consistency_score"], 0.80)
        self.assertEqual(consensus["senal_wallets"], "BUY")
        self.assertEqual(consensus["n_wallets_dirigidas"], 1)

    def test_equal_support_is_neutral(self):
        profiles = pd.DataFrame([
            {"wallet": "0xA", "accion": "BUY_POL", "horizonte": "1h", "winner_status": "winner", "consistency_score": 0.85},
            {"wallet": "0xB", "accion": "SELL_POL", "horizonte": "1h", "winner_status": "winner", "consistency_score": 0.85},
        ])
        consensus = calcular_consenso_wallets_1h(profiles)

        self.assertEqual(consensus["senal_wallets"], "NEUTRAL")
        self.assertEqual(consensus["confianza_wallets"], 0.0)


if __name__ == "__main__":
    unittest.main()
