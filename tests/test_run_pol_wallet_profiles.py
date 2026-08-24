import tempfile
import unittest
from pathlib import Path

import pandas as pd

from run_pol_wallet_profiles import ejecutar_flujo


WALLET = "0x1111111111111111111111111111111111111111"
T0 = pd.Timestamp("2026-08-20T00:00:00Z")


class MainFlowTests(unittest.TestCase):
    def test_main_flow_writes_report_and_returns_winners(self):
        raw = pd.DataFrame([
            {
                "timestamp": T0 + pd.Timedelta(hours=index), "bloque": index + 1,
                "hash_tx": f"0xbuy{index}", "wallet": WALLET, "direccion": "Buy",
                "pol_cantidad": 1.0, "usdc_cantidad": float(index + 1),
            }
            for index in range(4)
        ])
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "swaps_24h.parquet"
            raw.to_parquet(source, index=False)
            result, winners = ejecutar_flujo(
                parquet_path=source, output_dir=root / "output", min_decisiones=3,
            )

            self.assertTrue((result / "informe_wallets.html").is_file())
            self.assertEqual(len(winners), 1)
            self.assertEqual(winners.iloc[0]["accion"], "BUY_POL")


if __name__ == "__main__":
    unittest.main()
