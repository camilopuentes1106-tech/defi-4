import unittest
import tempfile
from pathlib import Path

import pandas as pd

from pol_rl_mdp import (
    construir_mdp_empirico,
    construir_observaciones_horarias,
    replay_historico,
    resolver_bellman_finito,
    simular_politica,
    tabla_politica,
    verificar_probabilidades,
)
from pol_rl_pipeline import ejecutar_rl_desde_snapshot
from pol_rl_rewards import acciones_admisibles


def _observaciones() -> pd.DataFrame:
    return pd.DataFrame([
        {
            "as_of": "2026-08-22T00:00:00Z", "regimen_mercado": "UP", "senal_wallets": "BUY",
            "precio_t": 100.0, "precio_t1": 101.0, "costo_gas_ratio": 0.001,
            "support_buy": 0.90, "support_sell": 0.10, "support_hold": 0.0,
        },
        {
            "as_of": "2026-08-22T01:00:00Z", "regimen_mercado": "FLAT", "senal_wallets": "NEUTRAL",
            "precio_t": 101.0, "precio_t1": 100.5, "costo_gas_ratio": 0.001,
            "support_buy": 0.0, "support_sell": 0.0, "support_hold": 0.0,
        },
        {
            "as_of": "2026-08-22T02:00:00Z", "regimen_mercado": "DOWN", "senal_wallets": "SELL",
            "precio_t": 100.5, "precio_t1": 99.0, "costo_gas_ratio": 0.001,
            "support_buy": 0.10, "support_sell": 0.90, "support_hold": 0.0,
        },
        {
            "as_of": "2026-08-22T03:00:00Z", "regimen_mercado": "UP", "senal_wallets": "HOLD",
            "precio_t": 99.0, "precio_t1": 100.0, "costo_gas_ratio": 0.001,
            "support_buy": 0.10, "support_sell": 0.10, "support_hold": 0.90,
        },
    ])


class EmpiricalMDPTests(unittest.TestCase):
    def test_mdp_has_valid_probabilities_and_only_admissible_actions(self):
        mdp = construir_mdp_empirico(_observaciones())
        checks = verificar_probabilidades(mdp)

        self.assertEqual(len(mdp.states), 24)
        self.assertTrue(checks["valida"].all())
        for state_index, state in enumerate(mdp.states):
            valid = acciones_admisibles(state[2])
            for action, matrix in mdp.transitions.items():
                expected = 1.0 if action in valid else 0.0
                self.assertAlmostEqual(float(matrix[state_index].sum()), expected)

    def test_bellman_policy_simulation_and_historical_replay(self):
        observations = _observaciones()
        mdp = construir_mdp_empirico(observations)
        solution = resolver_bellman_finito(mdp, horizon=4, gamma=0.99)
        policy = tabla_politica(mdp, solution)
        simulation = simular_politica(mdp, solution, seed=9)
        replay = replay_historico(observations, mdp, solution)

        self.assertEqual(solution.values.shape, (5, 24))
        self.assertEqual(len(policy), 24)
        self.assertEqual(len(simulation), 4)
        self.assertEqual(len(replay), 4)
        self.assertTrue((replay["wealth"] > 0).all())
        for _, row in policy.iterrows():
            self.assertIn(row["accion_recomendada"], acciones_admisibles(int(row["posicion"])))

    def test_hourly_observations_only_use_price_known_at_each_cut(self):
        start = pd.Timestamp("2026-08-22T00:00:00Z")
        swaps = pd.DataFrame([
            {
                "timestamp": start + pd.Timedelta(hours=hour), "hash_tx": f"0x{hour}",
                "precio_ejecutado": 100.0 + hour, "direccion": "BUY_POL",
                "gas_usdc": 0.01, "notional_usdc": 100.0,
            }
            for hour in range(6)
        ])
        ledger = pd.DataFrame([
            {
                "wallet": "0xwallet", "accion": "BUY_POL", "horizonte": "1h",
                "decision_time": start + pd.Timedelta(hours=hour),
                "forward_time": start + pd.Timedelta(hours=hour + 1),
                "retorno_neto": 0.02, "ganancia_neta_usdc": 1.0,
            }
            for hour in range(3)
        ])

        observations = construir_observaciones_horarias(swaps, ledger)

        self.assertGreaterEqual(len(observations), 4)
        self.assertEqual(observations.iloc[0]["precio_t"], 100.0)
        self.assertEqual(observations.iloc[0]["precio_t1"], 101.0)
        self.assertIn("senal_wallets", observations.columns)

    def test_reproducible_pipeline_exports_guide_artifacts(self):
        start = pd.Timestamp("2026-08-22T00:00:00Z")
        swaps = pd.DataFrame([
            {
                "timestamp": start + pd.Timedelta(hours=hour), "hash_tx": f"0x{hour}",
                "precio_ejecutado": 100.0 + hour, "direccion": "BUY_POL",
                "gas_usdc": 0.01, "notional_usdc": 100.0,
            }
            for hour in range(6)
        ])
        ledger = pd.DataFrame([
            {
                "wallet": "0xwallet", "accion": "BUY_POL", "horizonte": "1h",
                "decision_time": start + pd.Timedelta(hours=hour),
                "forward_time": start + pd.Timedelta(hours=hour + 1),
                "retorno_neto": 0.02, "ganancia_neta_usdc": 1.0,
            }
            for hour in range(3)
        ])
        profiles = pd.DataFrame([{
            "wallet": "0xwallet", "accion": "BUY_POL", "horizonte": "1h",
            "winner_status": "winner", "consistency_score": 0.9,
        }])
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "wallet_snapshot"
            root.mkdir()
            profiles.to_parquet(root / "perfiles_wallet.parquet", index=False)
            swaps.to_parquet(root / "swaps_logicos.parquet", index=False)
            ledger.to_parquet(root / "ledger_decisiones.parquet", index=False)

            result = ejecutar_rl_desde_snapshot(
                snapshot_dir=root, output_dir=Path(temporary) / "rl", horizon=4,
            )

            self.assertEqual(len(result.wallets_dirigidas), 1)
            self.assertTrue((result.output_dir / "manifest.json").is_file())
            self.assertTrue((result.output_dir / "politica_bellman.parquet").is_file())
            self.assertTrue((result.output_dir / "verificacion_transiciones.parquet").is_file())


if __name__ == "__main__":
    unittest.main()
