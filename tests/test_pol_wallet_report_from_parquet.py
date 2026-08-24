import importlib.util
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from pol_wallet_report_from_parquet import (
    canonicalizar_swaps_logicos,
    generar_informe_desde_parquet,
)


WALLET = "0x1111111111111111111111111111111111111111"
T0 = pd.Timestamp("2026-08-20T00:00:00Z")


class ParquetOnlyTests(unittest.TestCase):
    def test_legacy_buy_sell_parquet_is_normalized(self):
        raw = pd.DataFrame([{
            "timestamp": T0,
            "bloque": 1,
            "hash_tx": "0xbuy",
            "wallet": WALLET,
            "direccion": "Buy",
            "pol_cantidad": 2.0,
            "usdc_cantidad": 4.0,
        }])
        logical = canonicalizar_swaps_logicos(raw)

        self.assertEqual(logical.loc[0, "direccion"], "BUY_POL")
        self.assertEqual(logical.loc[0, "pol_delta"], 2.0)
        self.assertEqual(logical.loc[0, "usdc_delta"], -4.0)

    @unittest.skipUnless(importlib.util.find_spec("plotly"), "plotly no está instalado")
    def test_report_is_generated_from_one_parquet_without_network(self):
        raw = pd.DataFrame([
            {
                "timestamp": T0,
                "bloque": 1,
                "hash_tx": "0xbuy",
                "wallet": WALLET,
                "direccion": "Buy",
                "pol_cantidad": 2.0,
                "usdc_cantidad": 4.0,
            },
            {
                "timestamp": T0 + pd.Timedelta(hours=3),
                "bloque": 2,
                "hash_tx": "0xsell",
                "wallet": WALLET,
                "direccion": "Sell",
                "pol_cantidad": 2.0,
                "usdc_cantidad": 5.0,
            },
        ])
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "swaps_20260820T030000Z.parquet"
            raw.to_parquet(source, index=False)
            output = generar_informe_desde_parquet(
                parquet_path=source, output_dir=Path(temporary) / "report", export_png=False
            )

            self.assertTrue((output / "informe_wallets.html").is_file())
            self.assertTrue((output / "resumen_wallets.parquet").is_file())
            self.assertTrue((output / "ciclos_hold.parquet").is_file())

    @unittest.skipUnless(importlib.util.find_spec("plotly"), "plotly no está instalado")
    def test_pool_prices_from_parquet_can_identify_a_mature_winner(self):
        raw = pd.DataFrame([
            {
                "timestamp": T0 + pd.Timedelta(hours=index), "bloque": index + 1,
                "hash_tx": f"0xbuy{index}", "wallet": WALLET, "direccion": "Buy",
                "pol_cantidad": 1.0, "usdc_cantidad": float(index + 1),
            }
            for index in range(4)
        ])
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "swaps_20260820T030000Z.parquet"
            raw.to_parquet(source, index=False)
            output = generar_informe_desde_parquet(
                parquet_path=source, output_dir=Path(temporary) / "report", export_png=False
            )
            profiles = pd.read_parquet(output / "perfiles_wallet.parquet")
            profile = profiles[(profiles["accion"] == "BUY_POL") & (profiles["horizonte"] == "1h")].iloc[0]

            self.assertEqual(profile["n_decisiones"], 3)
            self.assertEqual(profile["winner_status"], "winner")


if __name__ == "__main__":
    unittest.main()
