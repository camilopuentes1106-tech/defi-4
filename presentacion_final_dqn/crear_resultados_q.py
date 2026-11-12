"""Resumen reproducible del Q-learning tabular del replay común."""
from pathlib import Path
import json
import numpy as np
from lib_rl.bellman import ACCIONES


def guardar_resultados_q(result, folder):
    model = result['modelos']['Q-learning']
    mdp = result['mdp']
    observations = result['observaciones']
    replay = result['resultados'].query("metodo == 'Q-learning'").sort_values('t')
    visits = model.visits
    by_state = visits.sum(axis=(0, 2))
    states_n = int(np.count_nonzero(by_state))
    state_time = int(np.count_nonzero(visits.sum(axis=2)))
    cells = int(np.count_nonzero(visits))
    start = observations.iloc[0]
    start_state = (start.regimen_mercado, start.senal_wallets, 0)
    start_idx = mdp.states.index(start_state)
    start_buy = int(visits[0,start_idx,ACCIONES.index('BUY_POL')])
    start_hold = int(visits[0,start_idx,ACCIONES.index('HOLD')])
    total = int(visits.sum())
    def state_label(st): return '/'.join([st[0], st[1], 'POL' if st[2] else 'USDC'])
    rows = [r'\begin{tabular}{lrr}',r'\toprule',r'Estado inicial & Buy & Hold \\',r'\midrule',
            f'{state_label(start_state)} & {start_buy} & {start_hold} ' + r'\\',
            r'\bottomrule',r'\end{tabular}']
    actions = [r'\begin{tabular}{cll}',r'\toprule',r'Etapa & Estado & Acción \\',r'\midrule']
    import ast
    for _, row in replay.head(4).iterrows():
        state = ast.literal_eval(row.estado) if isinstance(row.estado,str) else row.estado
        action = {'BUY_POL':'Buy','SELL_POL':'Sell','HOLD':'Hold'}[row.accion]
        actions.append(f'{int(row.t)} & {state_label(state)} & {action} '+r'\\')
    actions.extend([r'\bottomrule',r'\end{tabular}'])
    state_rows = [r'\begin{tabular}{lr}',r'\toprule',r'Estado (mercado / wallets / posición) & Visitas \\',r'\midrule']
    for state, count in zip(mdp.states,by_state):
        if count: state_rows.append(f'{state_label(state)} & {int(count):,}'.replace(',',r'\,')+r' \\')
    state_rows += [r'\midrule',f'Total & {total:,}'.replace(',',r'\,')+r' \\',r'\bottomrule',r'\end{tabular}']
    cmds = {'QEstadosVisitados': states_n, 'QEstadosTemporales':state_time,
            'QCeldasVisitadas': cells, 'QActualizaciones': f'{total:,}'.replace(',',r'\,'),
            'QTablaInicio':'\n'.join(rows), 'QTablaAcciones':'\n'.join(actions),
            'QTablaVisitas':'\n'.join(state_rows)}
    Path(folder,'resultados_q.tex').write_text('% Generado desde Q-learning del replay común.\n'+
        ''.join('\\newcommand{\\'+k+'}{'+str(v)+'}\n' for k,v in cmds.items()),encoding='utf8')
    data={'horas':len(observations),'estados_visitados':states_n,'estados_temporales':state_time,
          'celdas_visitadas':cells,'actualizaciones':total,'estado_inicial':state_label(start_state),
          'buy_inicial':start_buy,'hold_inicial':start_hold}
    Path(folder,'figuras','q_diagnostico.json').write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf8')
    return data
