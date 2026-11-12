import unittest
import numpy as np
import pandas as pd
from lib_rl.dqn import experimento_pol, indices_validos, auditar, comparar_replay_historico
from lib_rl.bellman import ACCIONES
from lib_rl.explicacion_dqn import analizar_decisiones_dqn, matriz_transiciones_regimen
from lib_rl.qlearning import entrenar_q_learning, replay_q_learning_historico
from lib_rl.bellman import resolver_bellman_finito, replay_historico

class TestDqnPOL(unittest.TestCase):
    def test_replay_comun_descarta_hora_incompleta(self):
        ts = pd.date_range("2026-01-01", periods=6, freq="h", tz="UTC")
        obs = pd.DataFrame({"as_of": ts, "regimen_mercado": "FLAT", "senal_wallets": "NEUTRAL",
                            "precio_t": 100., "precio_t1": 101., "support_buy": 0., "support_sell": 0.,
                            "support_hold": 0., "costo_gas_ratio": .001})
        swaps = pd.DataFrame({"timestamp": ts})
        replay = comparar_replay_historico(obs, swaps, episodios_dqn=4, episodios_tabla=10)
        self.assertEqual(len(replay["observaciones"]), 5)
        self.assertEqual(len(replay["resultados"]), 20)
        self.assertEqual(set(replay["resultados"].metodo), {"DQN", "Q-learning", "Bellman", "Buy and Hold"})
        for _, block in replay["resultados"].groupby("metodo"):
            self.assertEqual(block.as_of.tolist(), ts[:5].tolist())
            self.assertTrue(block.estado.iloc[0].endswith(", 0)"))
            self.assertEqual(block.t.tolist(), list(range(5)))
        frame = replay['observaciones']
        original_q = entrenar_q_learning(frame, replay['mdp'], episodes=10, seed=7)
        q_replay = replay_q_learning_historico(frame, replay['mdp'], original_q)
        actual_q = replay['resultados'].query("metodo == 'Q-learning'")
        self.assertEqual(q_replay.accion.tolist(), actual_q.accion.tolist())
        np.testing.assert_allclose(q_replay.wealth_economica, actual_q.wealth)
        original_bellman = resolver_bellman_finito(replay['mdp'], horizon=len(frame))
        bellman_replay = replay_historico(frame, replay['mdp'], original_bellman)
        actual_bellman = replay['resultados'].query("metodo == 'Bellman'")
        self.assertEqual(bellman_replay.accion.tolist(), actual_bellman.accion.tolist())
        np.testing.assert_allclose(100 * (1 + bellman_replay.recompensa_base).cumprod(), actual_bellman.wealth)

    def test_split_actions_and_terminal(self):
        ts = pd.date_range("2026-01-01", periods=25, freq="h", tz="UTC")
        obs = pd.DataFrame({"as_of": ts[:-1], "regimen_mercado": "FLAT", "senal_wallets": "NEUTRAL",
                            "precio_t": 100., "precio_t1": 101., "support_buy": 0., "support_sell": 0.,
                            "support_hold": 0., "confianza_wallets": 0.})
        swaps = pd.DataFrame({"timestamp": ts, "usdc_cantidad": 100., "gas_usdc": .1})
        result = experimento_pol(obs, swaps, episodios=20)
        self.assertLess(result["train"][-1].as_of.iloc[-1], result["validacion_bloques"][0].as_of.iloc[0])
        self.assertLess(result["validacion_bloques"][0].as_of.iloc[-1], result["test"][0].as_of.iloc[0])
        self.assertEqual({ACCIONES[i] for i in indices_validos(0)}, {"BUY_POL","HOLD"})
        self.assertEqual({ACCIONES[i] for i in indices_validos(1)}, {"SELL_POL","HOLD"})
        model = result["modelos"]["DQN"]
        analisis = analizar_decisiones_dqn(result)
        self.assertEqual(len(analisis['train']), 20 * 4)
        self.assertEqual(len(analisis['test']), 4)
        self.assertEqual(matriz_transiciones_regimen(analisis['train']).to_numpy().sum(), 20 * 3)
        self.assertEqual(matriz_transiciones_regimen(analisis['test']).to_numpy().sum(), 3)
        self.assertEqual(analisis['acciones']['Train'].sum(), 80)
        self.assertEqual(analisis['acciones']['Test'].sum(), 4)
        self.assertTrue((analisis['test']['Q_accion'] >= analisis['test']['Q_alternativa']).all())
        self.assertLess(analisis['train']['as_of'].max(), analisis['test']['as_of'].min())
        for _, row in analisis['train'].iterrows():
            self.assertIn(row.accion, {'BUY_POL', 'HOLD'} if row.posicion_antes == 0 else {'SELL_POL', 'HOLD'})
            if row.done:
                self.assertIsNone(row.estado_siguiente)
            else:
                self.assertEqual(row.estado_siguiente[2], row.posicion_despues)
        self.assertTrue(analisis['test'].iloc[-1].done)
        self.assertIsNone(analisis['test'].iloc[-1].estado_siguiente)
        self.assertTrue(any(z[4] == 1 and not np.any(z[3]) for z in model.memoria))
        audit = auditar(model)
        self.assertAlmostEqual(audit["target_pb"], audit["recompensa_pb"] + .99 * max(
            audit["Q_objetivo"][list(map(ACCIONES.index, audit["acciones_futuras_validas"]))]), places=4)
        for _, block in result["resultados"].groupby("metodo"):
            pos = 0
            for _, row in block.iterrows():
                self.assertIn(row.accion, {"BUY_POL", "HOLD"} if pos == 0 else {"SELL_POL","HOLD"})
                pos = int(row.posicion)

if __name__ == "__main__":
    unittest.main()
