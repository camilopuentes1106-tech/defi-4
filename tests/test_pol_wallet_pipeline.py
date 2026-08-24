import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from pol_wallet_pipeline import ejecutar_desde_parquet


WALLET = "0x1111111111111111111111111111111111111111"
T0 = pd.Timestamp("2026-08-20T00:00:00Z")


class UnifiedPipelineTests(unittest.TestCase):
    def test_parquet_mode_returns_normalized_winner_result(self):
        profiles = pd.DataFrame([{
            "wallet": WALLET,
            "accion": "BUY_POL",
            "horizonte": "1h",
            "winner_status": "winner",
            "consistency_score": 0.8,
            "pnl_neto_usdc": 2.0,
            "n_decisiones": 3,
        }])
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            snapshot = root / "output" / "snapshot"
            snapshot.mkdir(parents=True)
            (snapshot / "perfiles_wallet.parquet").touch()
            with patch("pol_wallet_pipeline.generar_informe_desde_parquet", return_value=snapshot), patch(
                "pol_wallet_pipeline.pd.read_parquet", return_value=profiles,
            ):
                result = ejecutar_desde_parquet(
                    parquet_path=root / "swaps_24h.parquet", output_dir=root / "output", min_decisiones=3,
                )

            self.assertEqual(result.snapshot_dir, snapshot)
            self.assertEqual(len(result.ganadoras), 1)
            self.assertEqual(result.ganadoras.iloc[0]["accion"], "BUY_POL")


if __name__ == "__main__":
    unittest.main()
