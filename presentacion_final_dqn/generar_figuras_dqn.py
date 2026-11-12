"""Regenera las figuras DQN desde cache/swaps_24h.parquet.

Ejecutar desde la raíz del repositorio:
    python presentacion_final_dqn/generar_figuras_dqn.py
"""
from pathlib import Path
import json
import hashlib
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lib_rl.wallets import analizar_wallets
from lib_rl.bellman import ejecutar_agente_bellman
from lib_rl.dqn import experimento_pol, resumen_resultados, auditar, comparar_replay_historico
from presentacion_final_dqn.crear_figuras_decisiones import guardar_decisiones
from presentacion_final_dqn.actualizar_resultados_base import guardar_resultados_base
from presentacion_final_dqn.crear_resultados_q import guardar_resultados_q

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent / "figuras"
OUT.mkdir(exist_ok=True)
raw = pd.read_parquet(ROOT / "cache" / "swaps_24h.parquet")
print("Swaps cache:", len(raw), flush=True)
res = analizar_wallets(raw, min_decisiones=3)
print("Perfiles:", len(res.perfiles), flush=True)
r = ejecutar_agente_bellman(swaps_logicos=res.swaps_logicos, ledger=res.ledger,
                           perfiles=res.perfiles, consistency_threshold=.8,
                           flat_band=.001, profile_weight=.25, horizon=None, gamma=.99)
guardar_resultados_base(r, OUT.parent)
r.observaciones.to_csv(OUT/'observaciones_horarias.csv', index=False)
ex = experimento_pol(r.observaciones, res.swaps_logicos, horizonte=4, episodios=600, seed=7)
guardar_decisiones(ex, OUT.parent)
dqn = ex["modelos"]["DQN"]
a = auditar(dqn)
palette = {'DQN':'#D77C1A', 'Q-learning':'#2F6B9A', 'Bellman':'#7954A2', 'Buy and Hold':'#64748b'}
plt.rcParams.update({'font.size':11, 'axes.titlesize':13, 'axes.labelsize':11,
                     'axes.spines.top':False, 'axes.spines.right':False,
                     'figure.facecolor':'white', 'axes.facecolor':'white'})

fig, axes = plt.subplots(1, 2, figsize=(11.5, 3.5))
log = dqn.registro
axes[0].plot(log.episodio, log.recompensa_pb, color='#2F6B9A', alpha=.16, lw=.8)
axes[0].plot(log.episodio, log.recompensa_pb.rolling(30,min_periods=1).mean(),
             color='#2F6B9A', lw=2, label='Media móvil: 30 episodios')
axes[0].axhline(0, color='#95a1ac', lw=.7)
axes[0].set(xlabel='Episodio', ylabel='Recompensa ajustada (pb)')
axes[0].legend(frameon=False, fontsize=9)
axes[1].plot(np.arange(1,len(dqn.perdidas)+1), dqn.perdidas, color='#D77C1A', alpha=.15, lw=.6)
axes[1].plot(np.arange(1,len(dqn.perdidas)+1),
             pd.Series(dqn.perdidas).rolling(100,min_periods=1).mean(), color='#D77C1A', lw=2,
             label='Media móvil: 100 actualizaciones')
axes[1].set(xlabel='Actualización', ylabel='Pérdida Huber')
axes[1].legend(frameon=False, fontsize=9)
fig.tight_layout()
fig.savefig(OUT/'dqn_entrenamiento.pdf', bbox_inches='tight')
plt.close(fig)

met = resumen_resultados(ex['resultados'])
fig, axes = plt.subplots(1, 2, figsize=(11.5, 3.5))
estilos = {
    'Buy and Hold': {'linewidth': 2, 'markersize': 5, 'marker': 'D'},
    'Bellman': {'linewidth': 2, 'markersize': 5, 'marker': '^'},
    'DQN': {'linewidth': 2, 'markersize': 5, 'marker': 'o'},
    'Q-learning': {'linewidth': 2, 'markersize': 5, 'marker': 's'},
}
for name in ('Buy and Hold', 'Bellman', 'DQN', 'Q-learning'):
    frame = ex['resultados'].query("metodo == @name").sort_values('t')
    axes[0].plot([0]+(frame.t+1).tolist(), [100]+frame.wealth.tolist(),
                 solid_capstyle='round', label=name, color=palette[name], **estilos[name])
axes[0].axhline(100, color='#95a1ac', ls=':', lw=1)
axes[0].set(xlabel='Hora del bloque de test', ylabel='Capital económico (inicial 100)')
axes[0].set_xticks(range(5))
handles, labels = axes[0].get_legend_handles_labels()
legend_order = [labels.index(m) for m in ('DQN', 'Q-learning', 'Bellman', 'Buy and Hold')]
axes[0].legend([handles[i] for i in legend_order], [labels[i] for i in legend_order],
               frameon=False, fontsize=9, loc='best')
vals=100*met['Retorno económico'].to_numpy()
axes[1].barh(met['Método'],vals,color=[palette[m] for m in met['Método']])
axes[1].axvline(0,color='#95a1ac',lw=.8)
axes[1].set(xlabel='Retorno económico neto (%)')
for i,v in enumerate(vals):
    axes[1].annotate(f'{v:+.2f}%', (v,i), xytext=(-4 if v>=0 else 4,0),
                     textcoords='offset points', va='center', ha='right' if v>=0 else 'left',
                     color='white', fontweight='bold', fontsize=9)
lo,hi=axes[1].get_xlim(); axes[1].set_xlim(lo-.015,hi+.015)
fig.tight_layout()
fig.savefig(OUT/'dqn_comparacion_test.pdf',bbox_inches='tight')
plt.close(fig)

base = resumen_resultados(ex['resultados']).set_index('Método')
stress = resumen_resultados(ex['sin_wallets']).set_index('Método')
fig, ax = plt.subplots(figsize=(9,3.5))
methods=list(base.index); y=np.arange(len(methods))
ax.barh(y+.18,100*base.loc[methods,'Retorno económico'],height=.34,label='Señal observada',color='#2F6B9A')
ax.barh(y-.18,100*stress.loc[methods,'Retorno económico'],height=.34,label='Señal neutral',color='#D77C1A')
ax.set_yticks(y,methods); ax.axvline(0,color='#95a1ac',lw=.8)
ax.set_xlabel('Retorno económico neto en el mismo test (%)')
ax.legend(frameon=False, loc='lower right', bbox_to_anchor=(1, 1.01),
          ncol=2, fontsize=9)
fig.tight_layout()
fig.savefig(OUT/'dqn_estres.pdf',bbox_inches='tight')
plt.close(fig)

valid=list(a['acciones_futuras_validas'])
all_actions=['BUY_POL','SELL_POL','HOLD']
qfuture=max(float(a['Q_objetivo'][all_actions.index(v)]) for v in valid)
details={
  'cache_sha256':hashlib.sha256((ROOT/'cache/swaps_24h.parquet').read_bytes()).hexdigest(),
  'ultima_transaccion_utc':str(pd.to_datetime(raw['timestamp'],utc=True).max()),
  'cache_filas':len(raw), 'observaciones':len(r.observaciones),
  'train_bloques':len(ex['train']), 'horizonte':4,
  'validacion':ex['validacion'].to_dict(orient='records'),
  'epsilon_seleccionado':ex['epsilon_elegido'],
  'test_inicio':str(ex['test'][0].as_of.iloc[0]),
  'test_fin_decision':str(ex['test'][0].as_of.iloc[-1]),
  'gas_ratio':ex['gas'], 'actualizaciones':dqn.actualizaciones,
  'sincronizaciones':dqn.sincronizaciones, 'experiencias':len(dqn.memoria),
  'auditoria':{'accion':a['accion'],'recompensa':a['recompensa_pb'],
      'q':a['Q_accion'],'q_futuro_valido':qfuture,'target':a['target_pb'], 'huber':a['huber'], 'done':a['done']},
  'decision_test': {
      k: (v.item() if isinstance(v, np.generic) else v)
      for k, v in ex['resultados'].query("metodo == 'DQN'").iloc[0]
          [['accion', 'Q_BUY_POL', 'Q_HOLD']].to_dict().items()
  },
  'test':met.to_dict(orient='records'), 'estres':resumen_resultados(ex['sin_wallets']).to_dict(orient='records')
}
with (OUT/'dqn_resultados.json').open('w',encoding='utf-8') as f: json.dump(details,f,ensure_ascii=False,indent=2,default=float)
def comando(nombre, valor):
    return "\\newcommand{\\" + nombre + "}{" + str(valor) + "}\n"

aud = details['auditoria']
commands = {
    'DQNExperiencias': details['experiencias'],
    'DQNActualizaciones': details['actualizaciones'],
    'DQNSincronizaciones': details['sincronizaciones'],
    'DQNAccion': aud['accion'].replace('_', r'\_'),
    'DQNRecompensa': f"{aud['recompensa']:.2f}",
    'DQNPredict': f"{aud['q']:.2f}",
    'DQNMaxFuturo': f"{aud['q_futuro_valido']:.2f}",
    'DQNTarget': f"{aud['target']:.2f}",
    'DQNHube': f"{aud['huber']:.3f}",
    'DQNEpsilon': f"{details['epsilon_seleccionado']:.2f}",
    'DQNValidacionA': f"{100*details['validacion'][0]['retorno_economico']:+.2f}",
    'DQNValidacionB': f"{100*details['validacion'][1]['retorno_economico']:+.2f}",
    'DQNDecisionTest': details['decision_test']['accion'].replace('_', r'\_'),
    'DQNQBuy': f"{details['decision_test']['Q_BUY_POL']:.2f}",
    'DQNQHold': f"{details['decision_test']['Q_HOLD']:.2f}",
    **{f'DQNTest{tipo}{sufijo}': f"{100 * fila[columna]:+.2f}"
       for sufijo, metodo in [('DQN','DQN'), ('Q','Q-learning'), ('Bellman','Bellman'), ('Hold','Buy and Hold')]
       for fila in details['test'] if fila['Método'] == metodo
       for tipo, columna in [('Ret','Retorno económico'), ('Draw','Máx. drawdown')]},
    'DQNTestInicio': pd.Timestamp(details['test_inicio']).strftime('%d-%b %H:%M'),
    'DQNTestFin': (pd.Timestamp(details['test_fin_decision']) + pd.Timedelta(hours=1)).strftime('%d-%b %H:%M'),
}
(OUT.parent/'resultados_dqn.tex').write_text(
    '% Generado desde figuras/dqn_resultados.json.\n' + ''.join(comando(k, v) for k, v in commands.items()),
    encoding='utf-8')
print(json.dumps(details,ensure_ascii=False,indent=2,default=float), flush=True)

print("Entrenando los cuatro métodos para el replay común...", flush=True)
replay_comun = comparar_replay_historico(
    r.observaciones, res.swaps_logicos, max_horas=24,
    episodios_dqn=600, episodios_tabla=2000, seed=7
)
guardar_resultados_q(replay_comun, OUT.parent)
frame_replay = replay_comun['resultados']
metricas_replay = replay_comun['metricas']
horas = len(replay_comun['observaciones'])
fig, ax = plt.subplots(figsize=(11.5, 4.6))
for i, name in enumerate(('Buy and Hold', 'Bellman', 'DQN', 'Q-learning')):
    frame = frame_replay.query("metodo == @name").sort_values('t')
    ax.plot([0]+(frame.t+1).tolist(), [100]+frame.wealth.tolist(),
            markevery=(i, 4), solid_capstyle='round', label=name,
            color=palette[name], **estilos[name])
ax.axhline(100, color='#95a1ac', ls=':', lw=1)
ax.set(xlabel='Horas transcurridas desde el inicio del replay',
       ylabel='Capital económico neto (inicial = 100)')
handles, labels = ax.get_legend_handles_labels()
order = [labels.index(m) for m in ('DQN', 'Q-learning', 'Bellman', 'Buy and Hold')]
ax.legend([handles[i] for i in order], [labels[i] for i in order],
          frameon=False, fontsize=10, loc='best')
ax.grid(alpha=.2)
fig.tight_layout()
fig.savefig(OUT/'replay_comun_pol.pdf', bbox_inches='tight')
plt.close(fig)
frame_replay.to_csv(OUT/'replay_comun_pol.csv', index=False)
inicio = replay_comun['observaciones'].as_of.iloc[0]
fin = replay_comun['observaciones'].as_of.iloc[-1] + pd.Timedelta(hours=1)
replay_info = {'horas_completas': horas, 'inicio_utc': str(inicio),
               'fin_utc': str(fin), 'metricas': metricas_replay.to_dict(orient='records'),
               'alcance': 'replay dentro de muestra; los modelos entrenan en el mismo período'}
(OUT/'replay_comun_pol.json').write_text(
    json.dumps(replay_info, ensure_ascii=False, indent=2, default=float), encoding='utf8'
)
replay_commands = {'DQNReplayHoras': horas,
                   'DQNReplayInicio': inicio.strftime('%d-%b %H:%M'),
                   'DQNReplayFin': fin.strftime('%d-%b %H:%M')}
(OUT.parent/'resultados_replay.tex').write_text(
    '% Generado desde figuras/replay_comun_pol.json.\n'
    + ''.join(comando(k,v) for k,v in replay_commands.items()), encoding='utf8'
)
print('Replay común completado:', horas, 'horas.', flush=True)
print(metricas_replay.to_string(index=False), flush=True)
