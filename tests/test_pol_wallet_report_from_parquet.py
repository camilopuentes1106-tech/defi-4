import importlib.util
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from pol_wallet_report_from_parquet import (
    canonicalizar_swaps_logicos,
    construir_ciclos_hold,
    construir_ledger_decisiones,
    construir_precios_por_minuto_desde_swaps,
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

    def test_minute_horizon_never_uses_a_swap_after_the_target_time(self):
        raw = pd.DataFrame([
            {
                "timestamp": T0 + pd.Timedelta(seconds=30), "bloque": 1, "hash_tx": "0xbuy",
                "wallet": WALLET, "direccion": "Buy", "pol_cantidad": 1.0, "usdc_cantidad": 1.0,
            },
            {
                "timestamp": T0 + pd.Timedelta(minutes=1, seconds=45), "bloque": 2, "hash_tx": "0xsell",
                "wallet": WALLET, "direccion": "Sell", "pol_cantidad": 1.0, "usdc_cantidad": 2.0,
            },
        ])
        logical = canonicalizar_swaps_logicos(raw)
        prices = construir_precios_por_minuto_desde_swaps(logical)
        ledger = construir_ledger_decisiones(
            logical, construir_ciclos_hold(logical), prices,
            as_of=T0 + pd.Timedelta(minutes=1, seconds=45),
        )

        buy_1m = ledger[(ledger["accion"] == "BUY_POL") & (ledger["horizonte"] == "1m")].iloc[0]
        self.assertEqual(buy_1m["maturity_status"], "COMPLETED")
        self.assertAlmostEqual(buy_1m["forward_price"], 1.0)

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

    @unittest.skipUnless(importlib.util.find_spec("plotly"), "plotly no está instalado")
    def test_report_accepts_walletview_table_for_position_horizon_ranking(self):
        raw = pd.DataFrame([{
            "timestamp": T0, "bloque": 1, "hash_tx": "0xbuy", "wallet": WALLET,
            "direccion": "Buy", "pol_cantidad": 1.0, "usdc_cantidad": 1.0,
        }])
        wallet_table = pd.DataFrame([{
            "wallet": WALLET, "direccion": "Buy", "n_swaps": 3,
            "pol_cantidad": 3.0, "usdc_cantidad": 3.0, "gas_usdc": 0.1, "usdc_neto": 2.9,
            "ganancia_neta_usdc_1m": 0.2, "ganancia_neta_usdc_5m": 0.3,
            "ganancia_neta_usdc_15m": 0.4, "ganancia_neta_usdc_1h": 0.5,
        }])
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "swaps_20260820T000000Z.parquet"
            raw.to_parquet(source, index=False)
            output = generar_informe_desde_parquet(
                parquet_path=source, output_dir=Path(temporary) / "report",
                tabla_wallets=wallet_table, export_png=False,
            )
            summary_profiles = pd.read_parquet(output / "perfiles_tabla_wallets.parquet")

        self.assertEqual(len(summary_profiles), 4)
        self.assertTrue(summary_profiles["es_candidata"].all())


if __name__ == "__main__":
    unittest.main()
