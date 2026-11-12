"""Genera cifras de Bellman desde el mismo procesamiento del cuaderno."""
from itertools import product
from pathlib import Path
import pandas as pd
from lib_rl.bellman import REGIMENES_MERCADO, SENALES_WALLETS, verificar_probabilidades


def guardar_resultados_base(rl, folder):
    folder = Path(folder)
    obs, policy, replay = rl.observaciones, rl.politica, rl.replay
    counts = obs.senal_wallets.value_counts()
    policy_counts = policy.accion_recomendada.value_counts()
    checks = verificar_probabilidades(rl.mdp)
    macros = {'ObservacionesRL': len(obs), 'SenalBuy': counts.get('BUY', 0),
              'SenalSell': counts.get('SELL', 0), 'SenalHold': counts.get('HOLD', 0),
              'SenalNeutral': counts.get('NEUTRAL', 0), 'TransicionesVerificadas': len(checks),
              'TransicionesValidas': 'sí' if checks.valida.all() else 'no',
              'PoliticaBuy': policy_counts.get('BUY_POL', 0),
              'PoliticaSell': policy_counts.get('SELL_POL', 0),
              'PoliticaHold': policy_counts.get('HOLD', 0)}

    def tabla_replay(data):
        rows=[r'\begin{tabular}{llcrrrrc}',r'\toprule',
              r'Hora UTC & R/W/X & Acción & $X_{t+1}$ & Base & Bono & Final & Riqueza \\',r'\midrule']
        for _, row in data.iterrows():
            state = str(row.estado).replace(' | ', '/').replace('posición=', '').replace('NEUTRAL', 'NEU')
            action = {'BUY_POL':'Buy','SELL_POL':'Sell','HOLD':'Hold'}[row.accion]
            vals = [f'{100*float(row[c]):+.4f}\\%' for c in ('recompensa_base','bonus_wallets','recompensa_final')]
            rows.append(f"{pd.Timestamp(row.as_of):%d %b %Hh} & {state} & {action} & {int(row.posicion_despues)} & " + ' & '.join(vals) + f" & {row.wealth:.2f} " + r'\\')
        return '\n'.join(rows+[r'\bottomrule',r'\end{tabular}'])
    macros['TablaReplayPrimera'] = tabla_replay(replay.iloc[:12])
    macros['TablaReplaySegunda'] = tabla_replay(replay.iloc[12:])
    states = list(product(REGIMENES_MERCADO, SENALES_WALLETS))
    index = {state:i for i,state in enumerate(rl.mdp.states)}
    def label(state): return state[0][0]+'/'+state[1][0]
    def table_matrix(start):
        subset=states[start:start+6]
        rows=[r'\begin{tabular}{lrrrrrr}', r'\toprule',
              'Origen & '+' & '.join(map(label,subset))+r' \\', r'\midrule']
        for state in states:
            si=index[(*state,0)]
            vals=[rl.mdp.transitions['HOLD'][si,index[(*target,0)]] for target in subset]
            rows.append(label(state)+' & '+' & '.join(f'{v:.3f}' for v in vals)+r' \\')
        return '\n'.join(rows+[r'\bottomrule',r'\end{tabular}'])
    macros['TablaMatrizTransicionesUno'] = table_matrix(0)
    macros['TablaMatrizTransicionesDos'] = table_matrix(6)
    (folder/'resultados_ejecucion.tex').write_text('% Generado con los swaps y estados actuales.\n'+
        ''.join('\\newcommand{\\'+k+'}{'+str(v)+'}\n' for k,v in macros.items()),encoding='utf8')
