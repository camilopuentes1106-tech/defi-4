import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from defi4.wallets.profiles import (
    DEFAULT_POOL_ADDRESS,
    canonicalizar_swaps_logicos,
    construir_ciclos_hold,
    construir_ledger_decisiones,
    construir_perfiles_wallet,
    construir_estado_rl,
    crear_figuras_perfiles,
    descargar_precios_horarios,
    exportar_snapshot_derivado,
    reconstruir_perfiles_desde_cache,
)


WALLET = "0x1111111111111111111111111111111111111111"
T0 = pd.Timestamp("2026-08-20T00:00:00Z")


def raw_swap(timestamp, tx_hash, pol_delta, usdc_delta, gas=0.0, log_index=0):
    quantity = abs(pol_delta)
    return {
        "timestamp": timestamp,
        "block_number": 1,
        "log_index": log_index,
        "hash_tx": tx_hash,
        "wallet": WALLET,
        "pol_delta": pol_delta,
        "usdc_delta": usdc_delta,
        "pol_cantidad": quantity,
        "usdc_cantidad": abs(usdc_delta),
        "precio_ejecutado": abs(usdc_delta) / quantity if quantity else np.nan,
        "gas_usdc": gas,
    }


class LogicalSwapTests(unittest.TestCase):
    def test_default_pool_address_is_available_for_notebooks(self):
        self.assertEqual(DEFAULT_POOL_ADDRESS, "0xA374094527e1673A86dE625aa59517c5dE346d32")

    def test_multihop_is_netted_and_gas_is_not_duplicated(self):
        swaps = pd.DataFrame([
            raw_swap(T0, "0xaaa", 2, -4, gas=0.1, log_index=1),
            raw_swap(T0, "0xaaa", 1, -2, gas=0.1, log_index=2),
            raw_swap(T0 + pd.Timedelta(hours=1), "0xbbb", 2, -2, log_index=3),
            raw_swap(T0 + pd.Timedelta(hours=1), "0xbbb", -2, 2, log_index=4),
        ])
        logical = canonicalizar_swaps_logicos(swaps)

        buy = logical.loc[logical["hash_tx"] == "0xaaa"].iloc[0]
        self.assertEqual(buy["direccion"], "BUY_POL")
        self.assertEqual(buy["event_count"], 2)
        self.assertAlmostEqual(buy["pol_cantidad"], 3)
        self.assertAlmostEqual(buy["notional_usdc"], 6)
        self.assertAlmostEqual(buy["gas_usdc"], 0.1)
        self.assertEqual(logical.loc[logical["hash_tx"] == "0xbbb", "direccion"].item(), "AMBIGUOUS")

    def test_original_cache_format_is_accepted_without_redownloading(self):
        legacy = pd.DataFrame([{
            "timestamp": T0, "bloque": 1, "hash_tx": "0xlegacy", "wallet": WALLET,
            "direccion": "Buy", "pol_cantidad": 2.0, "usdc_cantidad": 4.0,
            "precio_ejecutado": 2.0, "gas_usdc": 0.1,
        }])
        logical = canonicalizar_swaps_logicos(legacy)

        self.assertEqual(logical.iloc[0]["direccion"], "BUY_POL")
        self.assertAlmostEqual(logical.iloc[0]["pol_delta"], 2.0)
        self.assertAlmostEqual(logical.iloc[0]["usdc_delta"], -4.0)

    def test_fifo_partial_closes_and_censored_inventory(self):
        logical = pd.DataFrame([
            {"timestamp": T0, "hash_tx": "0xbuy", "wallet": WALLET, "direccion": "BUY_POL", "pol_cantidad": 10.0, "precio_ejecutado": 1.0, "gas_usdc": 1.0, "pol_delta": 10.0, "notional_usdc": 10.0},
            {"timestamp": T0 + pd.Timedelta(hours=2), "hash_tx": "0xsell1", "wallet": WALLET, "direccion": "SELL_POL", "pol_cantidad": 4.0, "precio_ejecutado": 1.5, "gas_usdc": 0.4, "pol_delta": -4.0, "notional_usdc": 6.0},
            {"timestamp": T0 + pd.Timedelta(hours=5), "hash_tx": "0xsell2", "wallet": WALLET, "direccion": "SELL_POL", "pol_cantidad": 8.0, "precio_ejecutado": 1.25, "gas_usdc": 0.8, "pol_delta": -8.0, "notional_usdc": 10.0},
        ])
        cycles = construir_ciclos_hold(logical)
        closed = cycles[cycles["cycle_status"] == "CLOSED"].reset_index(drop=True)
        censored = cycles[cycles["cycle_status"] == "CENSORED_PREEXISTING"]

        self.assertEqual(len(closed), 2)
        self.assertAlmostEqual(closed.loc[0, "pol_cantidad"], 4.0)
        self.assertAlmostEqual(closed.loc[0, "pnl_realizado_usdc"], 1.2)
        self.assertAlmostEqual(closed.loc[1, "pol_cantidad"], 6.0)
        self.assertAlmostEqual(closed.loc[1, "pnl_realizado_usdc"], 0.3)
        self.assertEqual(len(censored), 1)
        self.assertAlmostEqual(censored.iloc[0]["pol_cantidad"], 2.0)


class LedgerAndProfileTests(unittest.TestCase):
    def setUp(self):
        self.logical = pd.DataFrame([
            {"timestamp": T0, "hash_tx": "0xbuy", "wallet": WALLET, "direccion": "BUY_POL", "pol_cantidad": 10.0, "precio_ejecutado": 1.0, "gas_usdc": 0.1, "pol_delta": 10.0, "notional_usdc": 10.0},
            {"timestamp": T0 + pd.Timedelta(hours=5), "hash_tx": "0xsell", "wallet": WALLET, "direccion": "SELL_POL", "pol_cantidad": 10.0, "precio_ejecutado": 1.3, "gas_usdc": 0.1, "pol_delta": -10.0, "notional_usdc": 13.0},
        ])
        self.cycles = construir_ciclos_hold(self.logical)
        self.prices = pd.DataFrame(
            {"Close": [1.0, 1.2, 1.3, 1.4]},
            index=[T0, T0 + pd.Timedelta(hours=1), T0 + pd.Timedelta(hours=4), T0 + pd.Timedelta(hours=5)],
        )

    def test_forward_signs_hold_eligibility_and_pending_short_horizons(self):
        ledger = construir_ledger_decisiones(
            self.logical, self.cycles, self.prices, as_of=T0 + pd.Timedelta(hours=5)
        )
        buy_1h = ledger[(ledger["accion"] == "BUY_POL") & (ledger["horizonte"] == "1h")].iloc[0]
        sell_1h = ledger[(ledger["accion"] == "SELL_POL") & (ledger["horizonte"] == "1h")].iloc[0]
        hold_1h = ledger[(ledger["accion"] == "HOLD") & (ledger["horizonte"] == "1h")].iloc[0]

        self.assertEqual(buy_1h["maturity_status"], "COMPLETED")
        self.assertAlmostEqual(buy_1h["retorno_neto"], 0.19)
        self.assertEqual(sell_1h["maturity_status"], "PENDING")
        self.assertAlmostEqual(hold_1h["retorno_neto"], 0.2)
        self.assertEqual(set(ledger["horizonte"]), {"1m", "5m", "15m", "1h"})
        self.assertTrue((ledger[ledger["accion"] == "SELL_POL"]["maturity_status"] == "PENDING").all())

    def test_winner_requires_three_completed_decisions_and_state_excludes_future_trade(self):
        completed = pd.DataFrame([
            {
                "decision_id": f"d{index}", "wallet": WALLET, "accion": "BUY_POL", "horizonte": "1h",
                "maturity_status": "COMPLETED", "ganancia_neta_usdc": 1.0,
                "retorno_neto": 0.1, "decision_time": T0,
            }
            for index in range(3)
        ] + [{
            "decision_id": "pending", "wallet": WALLET, "accion": "SELL_POL", "horizonte": "1h",
            "maturity_status": "PENDING", "ganancia_neta_usdc": np.nan,
            "retorno_neto": np.nan, "decision_time": T0,
        }])
        profiles = construir_perfiles_wallet(completed, self.logical, self.cycles)
        winner = profiles[profiles["accion"] == "BUY_POL"].iloc[0]
        self.assertEqual(winner["winner_status"], "winner")
        self.assertAlmostEqual(winner["consistency_score"], 0.86)
        self.assertEqual(profiles[profiles["accion"] == "SELL_POL"].iloc[0]["winner_status"], "pending")

        future = pd.concat([self.logical, pd.DataFrame([{
            "timestamp": T0 + pd.Timedelta(hours=2), "hash_tx": "0xfuture", "wallet": WALLET,
            "direccion": "BUY_POL", "pol_cantidad": 2.0, "precio_ejecutado": 1.0,
            "gas_usdc": 0.0, "pol_delta": 2.0, "notional_usdc": 2.0,
        }])], ignore_index=True)
        state = construir_estado_rl(future, self.cycles, profiles, as_of=T0 + pd.Timedelta(minutes=10), ventanas=["15m"])
        self.assertEqual(state.iloc[0]["n_swaps"], 1)

    @patch("yfinance.download")
    def test_price_download_falls_back_to_matic_when_pol_is_empty(self, download_mock):
        fallback = pd.DataFrame({"Close": [1.0, 1.1]}, index=[T0, T0 + pd.Timedelta(hours=1)])
        download_mock.side_effect = [pd.DataFrame(), fallback]
        prices = descargar_precios_horarios(as_of=T0 + pd.Timedelta(days=1), lookback_hours=24)

        self.assertEqual(download_mock.call_args_list[0].args[0], "POL-USD")
        self.assertEqual(download_mock.call_args_list[1].args[0], "MATIC-USD")
        self.assertEqual(download_mock.call_args_list[0].kwargs["interval"], "1m")
        self.assertEqual(prices["source_ticker"].iloc[0], "MATIC-USD")


@unittest.skipUnless(importlib.util.find_spec("plotly"), "plotly no está instalado")
class VisualExportTests(unittest.TestCase):
    def test_rebuild_from_cached_swaps_never_calls_alchemy(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw_dir = root / "raw"
            raw_dir.mkdir()
            pd.DataFrame([raw_swap(T0, "0xcached", 1, -1, log_index=1)]).to_parquet(
                raw_dir / "swaps_20260820T000000Z.parquet", index=False
            )
            derived = reconstruir_perfiles_desde_cache(
                raw_cache_dir=raw_dir, output_dir=root / "derived", as_of=T0,
                actualizar_precios=False, export_png=False,
            )
            self.assertTrue((derived / "swaps_logicos.parquet").is_file())

    def test_figures_and_html_report_are_exported(self):
        logical = pd.DataFrame([
            {"timestamp": T0, "hash_tx": "0xb", "wallet": WALLET, "direccion": "BUY_POL", "pol_cantidad": 1.0, "precio_ejecutado": 1.0, "gas_usdc": 0.0, "pol_delta": 1.0, "notional_usdc": 1.0},
            {"timestamp": T0 + pd.Timedelta(hours=2), "hash_tx": "0xs", "wallet": WALLET, "direccion": "SELL_POL", "pol_cantidad": 1.0, "precio_ejecutado": 1.2, "gas_usdc": 0.0, "pol_delta": -1.0, "notional_usdc": 1.2},
        ])
        cycles = construir_ciclos_hold(logical)
        ledger = pd.DataFrame([{
            "decision_id": "d", "wallet": WALLET, "accion": "BUY_POL", "horizonte": "1h",
            "maturity_status": "COMPLETED", "ganancia_neta_usdc": 0.2, "retorno_neto": 0.2,
            "decision_time": T0,
        }])
        profiles = construir_perfiles_wallet(ledger, logical, cycles, min_decisiones=1)
        figures = crear_figuras_perfiles(profiles, ledger, cycles, logical)
        self.assertEqual(set(figures), {"ranking", "mapa_retorno", "matriz", "ciclos_hold", "comportamiento"})
        with tempfile.TemporaryDirectory() as temporary:
            from plotly.basedatatypes import BaseFigure

            def write_image_stub(_, file, **__):
                Path(file).touch()

            with patch.object(BaseFigure, "write_image", new=write_image_stub):
                root = exportar_snapshot_derivado(
                    swaps_logicos=logical, ciclos_hold=cycles, ledger=ledger, perfiles=profiles,
                    estado_rl=pd.DataFrame(), output_dir=temporary, as_of=T0, export_png=True,
                )
            self.assertTrue((root / "informe_wallets.html").is_file())
            self.assertTrue((root / "manifest.json").is_file())
            self.assertEqual(len(list(root.glob("*.html"))), 6)
            self.assertEqual(len(list(root.glob("*.png"))), 5)

    def test_html_snapshot_is_retained_when_kaleido_fails(self):
        logical = pd.DataFrame([
            {"timestamp": T0, "hash_tx": "0xb", "wallet": WALLET, "direccion": "BUY_POL", "pol_cantidad": 1.0, "precio_ejecutado": 1.0, "gas_usdc": 0.0, "pol_delta": 1.0, "notional_usdc": 1.0},
        ])
        cycles = construir_ciclos_hold(logical)
        ledger = pd.DataFrame([{
            "decision_id": "d", "wallet": WALLET, "accion": "BUY_POL", "horizonte": "1h",
            "maturity_status": "PENDING", "ganancia_neta_usdc": np.nan, "retorno_neto": np.nan,
            "decision_time": T0,
        }])
        profiles = construir_perfiles_wallet(ledger, logical, cycles)
        from plotly.basedatatypes import BaseFigure

        with tempfile.TemporaryDirectory() as temporary:
            with patch.object(BaseFigure, "write_image", side_effect=ValueError("Kaleido no instalado")):
                root = exportar_snapshot_derivado(
                    swaps_logicos=logical, ciclos_hold=cycles, ledger=ledger, perfiles=profiles,
                    estado_rl=pd.DataFrame(), output_dir=temporary, as_of=T0, export_png=True,
                )
            manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
            self.assertTrue((root / "informe_wallets.html").is_file())
            self.assertEqual(len(manifest["png_export_errors"]), 5)


if __name__ == "__main__":
    unittest.main()
