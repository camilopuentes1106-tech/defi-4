import unittest

from defi4.model.rewards import (
    acciones_admisibles,
    calcular_recompensa_con_perfiles,
    posicion_despues,
)


class RewardTests(unittest.TestCase):
    def test_buy_reward_is_financial_return_plus_bonus_only_when_buy_has_support(self):
        reward = calcular_recompensa_con_perfiles(
            posicion=0,
            accion="BUY_POL",
            precio_t=100.0,
            precio_t1=101.0,
            costo_gas_ratio=0.001,
            support_wallets={"support_buy": 0.9, "support_sell": 0.1, "support_hold": 0.0},
        )

        self.assertAlmostEqual(reward["reward_base"], 0.009)
        self.assertAlmostEqual(reward["wallet_confidence"], 0.8)
        self.assertAlmostEqual(reward["advantage"], 0.009)
        self.assertAlmostEqual(reward["profile_bonus"], 0.0018)
        self.assertAlmostEqual(reward["reward_final"], 0.0108)

    def test_sell_receives_credit_when_it_avoids_a_price_drop_and_wallets_support_it(self):
        reward = calcular_recompensa_con_perfiles(
            posicion=1,
            accion="SELL_POL",
            precio_t=100.0,
            precio_t1=99.0,
            costo_gas_ratio=0.001,
            support_wallets={"support_buy": 0.1, "support_sell": 0.9, "support_hold": 0.0},
        )

        self.assertAlmostEqual(reward["reward_base"], -0.001)
        self.assertAlmostEqual(reward["advantage"], 0.009)
        self.assertGreater(reward["reward_final"], reward["reward_base"])

    def test_invalid_actions_are_not_available_for_the_current_position(self):
        self.assertEqual(acciones_admisibles(0), ("BUY_POL", "HOLD"))
        self.assertEqual(posicion_despues(1, "SELL_POL"), 0)
        with self.assertRaises(ValueError):
            calcular_recompensa_con_perfiles(
                posicion=0, accion="SELL_POL", precio_t=100.0, precio_t1=99.0,
                costo_gas_ratio=0.001,
            )


if __name__ == "__main__":
    unittest.main()
