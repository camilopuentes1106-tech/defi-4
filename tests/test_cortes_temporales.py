import unittest
import pandas as pd
from pandas.testing import assert_frame_equal

from lib_rl.wallets import (canonicalizar_swaps_logicos, construir_ciclos_hold,
    construir_ledger_decisiones, construir_perfiles_causales_1h, _precios_por_evento)
from lib_rl.bellman import construir_observaciones_horarias


class TestDisponibilidadTemporal(unittest.TestCase):
    def swaps(self):
        times = ['2026-01-01T00:00:05Z', '2026-01-01T00:10:05Z',
                 '2026-01-01T00:20:05Z', '2026-01-01T01:30:00Z',
                 '2026-01-01T02:30:00Z', '2026-01-01T03:00:00Z']
        return pd.DataFrame({'timestamp': pd.to_datetime(times, utc=True),
            'hash_tx': [f'tx{i}' for i in range(len(times))], 'wallet': ['w']*len(times),
            'direccion': ['Buy','Buy','Buy','Buy','Sell','Buy'],
            'pol_cantidad':[1.,1.,1.,1.,3.,1.], 'usdc_cantidad':[1.,1.,1.,2.,6.,2.],
            'gas_usdc':[.01,.01,.01,.01,.5,.01], 'gas_pol':[0.]*6})

    def ledger(self, data):
        c = canonicalizar_swaps_logicos(data)
        return construir_ledger_decisiones(c, construir_ciclos_hold(c), _precios_por_evento(c))

    def test_future_close_does_not_change_previous_wallet_profile(self):
        raw = self.swaps()
        cutoff = pd.Timestamp('2026-01-01T01:30:00Z')
        full = self.ledger(raw)
        prefix = self.ledger(raw[raw.timestamp <= cutoff])
        p1 = construir_perfiles_causales_1h(full, as_of=cutoff)
        p2 = construir_perfiles_causales_1h(prefix, as_of=cutoff)
        assert_frame_equal(p1, p2)
        self.assertFalse((p1.accion == 'HOLD').any())
        later = construir_perfiles_causales_1h(full, as_of='2026-01-01T02:30:00Z')
        self.assertTrue((later.accion == 'HOLD').any())

    def test_exact_event_price_and_unfinished_returns(self):
        swaps = canonicalizar_swaps_logicos(self.swaps())
        prices = _precios_por_evento(swaps)
        self.assertEqual(prices.asof(pd.Timestamp('2026-01-01T01:29:59Z')), 1.)
        ledger = construir_ledger_decisiones(swaps, construir_ciclos_hold(swaps), prices)
        last = ledger[(ledger.decision_time == swaps.timestamp.max()) & (ledger.accion == 'BUY_POL')]
        self.assertTrue(last.retorno_neto.isna().all())
        self.assertFalse(last.es_madura.any())
        same_minute = pd.DataFrame({'timestamp':pd.to_datetime(['2026-01-01T00:20:05Z',
                              '2026-01-01T00:20:50Z'],utc=True), 'precio_ejecutado':[1.,9.]})
        self.assertEqual(_precios_por_evento(same_minute).asof(pd.Timestamp('2026-01-01T00:20:10Z')), 1.)

    def test_incomplete_hour_is_excluded(self):
        swaps = canonicalizar_swaps_logicos(self.swaps())
        ledger = self.ledger(self.swaps())
        obs = construir_observaciones_horarias(swaps, ledger)
        self.assertTrue((obs.as_of + pd.Timedelta(hours=1) <= swaps.timestamp.max()).all())
        self.assertNotIn(swaps.timestamp.max(), obs.as_of.tolist())

    def test_legacy_hold_ledger_is_rejected(self):
        ledger = self.ledger(self.swaps()).drop(columns='disponible_desde')
        with self.assertRaisesRegex(ValueError, 'reconstruye el ledger'):
            construir_perfiles_causales_1h(ledger, as_of='2026-01-01T03:00:00Z')

    def test_equal_timestamps_are_ordered_identically_with_future_rows(self):
        swaps = self.swaps()
        swaps.loc[1:2, 'timestamp'] = swaps.loc[0, 'timestamp']
        prefix = canonicalizar_swaps_logicos(swaps.iloc[:4])
        full = canonicalizar_swaps_logicos(swaps)
        self.assertEqual(prefix.hash_tx.tolist(), full.iloc[:4].hash_tx.tolist())
        self.assertEqual(_precios_por_evento(prefix).iloc[0], _precios_por_evento(full).iloc[0])

if __name__ == '__main__':
    unittest.main()
