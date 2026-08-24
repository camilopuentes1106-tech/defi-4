import unittest
from unittest.mock import Mock, patch

import pandas as pd

from pol_dune_pipeline import (
    MIGRATION_START,
    DuneApi,
    MonthRange,
    RESULT_PAGE_SIZE,
    WRAPPED_POL_ADDRESS,
    candidate_wallets_sql,
    dex_trades_sql,
    gas_fees_sql,
    month_ranges,
    prices_hour_sql,
    selected_wallets,
    wallet_batches,
)


class QueryConstructionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.period = MonthRange(
            pd.Timestamp("2024-09-04T00:00:00Z"),
            pd.Timestamp("2024-10-01T00:00:00Z"),
        )

    def test_period_is_split_at_utc_month_boundaries(self) -> None:
        ranges = list(month_ranges(MIGRATION_START, pd.Timestamp("2024-11-02T00:00:00Z")))
        self.assertEqual(
            [item.partition for item in ranges],
            ["year=2024/month=09", "year=2024/month=10", "year=2024/month=11"],
        )
        self.assertEqual(ranges[0].start, MIGRATION_START)
        self.assertEqual(ranges[-1].end, pd.Timestamp("2024-11-02T00:00:00Z"))

    def test_dex_query_has_chain_time_and_contract_filters(self) -> None:
        sql = dex_trades_sql(self.period)
        self.assertIn("FROM dex.trades", sql)
        self.assertIn("d.blockchain = 'polygon'", sql)
        self.assertIn(WRAPPED_POL_ADDRESS, sql)
        self.assertIn("d.block_month >= DATE '2024-09-01'", sql)
        self.assertIn("d.block_time < TIMESTAMP '2024-10-01 00:00:00'", sql)

    def test_gas_query_deduplicates_hashes_before_the_join(self) -> None:
        sql = gas_fees_sql(self.period)
        self.assertIn("SELECT DISTINCT d.tx_hash", sql)
        self.assertIn("FROM gas.fees AS g", sql)
        self.assertIn("p.tx_hash = g.tx_hash", sql)

    def test_price_query_uses_contract_address_not_symbol(self) -> None:
        sql = prices_hour_sql(self.period)
        self.assertIn("FROM prices.hour AS p", sql)
        self.assertIn(f"p.contract_address = {WRAPPED_POL_ADDRESS}", sql)
        self.assertNotIn("p.symbol =", sql)

    def test_candidate_query_stratifies_without_adding_multihop_volume(self) -> None:
        sql = candidate_wallets_sql(self.period, wallets_per_quintile=2)
        self.assertIn("max(coalesce(d.amount_usd, 0)) AS tx_volume_proxy_usd", sql)
        self.assertIn("ntile(5) OVER (PARTITION BY month ORDER BY volume_proxy_usd)", sql)
        self.assertIn("WHERE sample_rank <= 2", sql)

    def test_wallet_filter_is_present_only_for_a_valid_cohort(self) -> None:
        wallet = "0x1111111111111111111111111111111111111111"
        self.assertIn(f"d.tx_from IN ({wallet})", dex_trades_sql(self.period, [wallet]))
        with self.assertRaises(ValueError):
            gas_fees_sql(self.period, ["not-a-wallet"])

    def test_selected_wallets_and_batches_are_normalized(self) -> None:
        wallets = selected_wallets([
            {"wallet": "0x1111111111111111111111111111111111111111"},
            {"wallet": "0x2222222222222222222222222222222222222222"},
        ])
        self.assertEqual(list(wallet_batches(wallets, 1)), [[wallets[0]], [wallets[1]]])
        with self.assertRaises(ValueError):
            list(wallet_batches(wallets, 0))


class CurlTransportTests(unittest.TestCase):
    @patch("pol_dune_pipeline.subprocess.run")
    @patch("pol_dune_pipeline.shutil.which", return_value="curl.exe")
    def test_curl_transport_encodes_result_pagination(self, _, run_mock: Mock) -> None:
        run_mock.return_value = Mock(
            returncode=0,
            stdout='{"result":{"rows":[]},"next_offset":10000}',
            stderr="",
        )
        client = DuneApi("test-key", transport="curl")
        result = client._request(
            "GET", "/execution/test/results", params={"limit": 10000, "offset": 0}
        )

        command = run_mock.call_args.args[0]
        self.assertEqual(result["next_offset"], 10000)
        self.assertIn("curl.exe", command)
        self.assertIn("https://api.dune.com/api/v1/execution/test/results?limit=10000&offset=0", command)
        self.assertNotIn("SSLKEYLOGFILE", run_mock.call_args.kwargs["env"])


class NodeTransportTests(unittest.TestCase):
    @patch("pol_dune_pipeline.subprocess.run")
    @patch("pol_dune_pipeline.shutil.which", return_value="node.exe")
    def test_node_transport_uses_system_ca_and_stdin(self, _, run_mock: Mock) -> None:
        run_mock.return_value = Mock(
            returncode=0,
            stdout='{"status":200,"body":"{\\"execution_id\\":\\"test\\"}"}',
            stderr="",
        )
        client = DuneApi("test-key", transport="node")
        result = client._request("POST", "/sql/execute", json={"sql": "SELECT 1"})

        command = run_mock.call_args.args[0]
        request = run_mock.call_args.kwargs["input"]
        self.assertEqual(result["execution_id"], "test")
        self.assertEqual(command[:3], ["node.exe", "--use-system-ca", "-e"])
        self.assertIn('"url":"https://api.dune.com/api/v1/sql/execute"', request)


class PaginationTests(unittest.TestCase):
    def test_pages_are_persistable_without_accumulating_rows(self) -> None:
        client = DuneApi("test-key")
        client._request = Mock(side_effect=[
            {"execution_id": "execution"},
            {"state": "QUERY_STATE_COMPLETED"},
            {"execution_id": "execution", "result": {"rows": [{"row": 1}]}, "next_offset": 1},
            {"execution_id": "execution", "result": {"rows": [{"row": 2}]}, "next_offset": None},
        ])
        pages = []
        result = client.execute_sql(
            "SELECT 1",
            on_page=lambda execution_id, rows, number, _: pages.append(
                (execution_id, rows, number)
            ),
        )

        self.assertEqual(result.rows, [])
        self.assertEqual(result.row_count, 2)
        self.assertEqual(pages, [
            ("execution", [{"row": 1}], 1),
            ("execution", [{"row": 2}], 2),
        ])
        self.assertEqual(client._request.call_args_list[2].kwargs["params"]["limit"], RESULT_PAGE_SIZE)


if __name__ == "__main__":
    unittest.main()
