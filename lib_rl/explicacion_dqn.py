"""Descripción de decisiones y transiciones del DQN en train y test."""
import ast

import numpy as np
import pandas as pd
from .bellman import ACCIONES, REGIMENES_MERCADO, acciones_admisibles


def analizar_decisiones_dqn(experimento):
    """Train contiene todas las acciones efectivas (incluida exploración).

    Test contiene la política final congelada aplicada al único bloque reservado.
    Las transiciones no cruzan límites de episodio y no se inventa un estado
    siguiente en el último paso; el cambio de posición sí se observa en él.
    """
    modelo = experimento['modelos']['DQN']
    if not hasattr(modelo, 'experiencias'):
        raise ValueError('Vuelve a ejecutar el entrenamiento con la versión actualizada del módulo.')
    train = modelo.experiencias.copy()
    test_obs = experimento['test'][0].reset_index(drop=True)
    source = experimento['resultados'].query("metodo == 'DQN'").sort_values('t')
    rows = []
    for _, r in source.iterrows():
        t = int(r.t)
        estado = ast.literal_eval(r.estado) if isinstance(r.estado, str) else tuple(r.estado)
        terminal = t == len(test_obs) - 1
        prox = None if terminal else (test_obs.iloc[t+1].regimen_mercado,
                                       test_obs.iloc[t+1].senal_wallets, int(r.posicion))
        validas = acciones_admisibles(int(estado[2]))
        q_validas = {a: float(r['Q_'+a]) for a in validas}
        rows.append({'episodio': 1, 'bloque': 1, 't': t, 'as_of': r.as_of,
                     'estado': estado, 'accion': r.accion, 'posicion_antes': int(estado[2]),
                     'posicion_despues': int(r.posicion), 'estado_siguiente': prox,
                     'done': terminal, 'exploracion': False, 'epsilon': 0.,
                     'acciones_validas': ', '.join(validas),
                     'Q_accion': q_validas[r.accion],
                     'Q_alternativa': next((v for a, v in q_validas.items() if a != r.accion), np.nan)})
    test = pd.DataFrame(rows)
    counts = pd.DataFrame({etapa: frame.accion.value_counts().reindex(ACCIONES, fill_value=0)
                           for etapa, frame in [('Train', train), ('Test', test)]})
    return {'train': train, 'test': test, 'acciones': counts}


def matriz_transiciones_regimen(decisiones):
    """Número de transiciones del componente régimen del estado completo."""
    matriz = pd.DataFrame(0, index=REGIMENES_MERCADO, columns=REGIMENES_MERCADO)
    for _, fila in decisiones.iterrows():
        if not fila.done:
            matriz.loc[fila.estado[0], fila.estado_siguiente[0]] += 1
    return matriz


def graficar_decisiones_dqn(analisis):
    """Dos gráficos: acciones relativas y transición del régimen; conteos en etiquetas."""
    import matplotlib.pyplot as plt
    colors = ['#d77c1a', '#7954a2', '#2f6b9a']
    fig_acc, axes = plt.subplots(1, 2, figsize=(11, 3.7))
    for ax, etapa in zip(axes, ('Train', 'Test')):
        counts = analisis['acciones'][etapa]
        total = int(counts.sum())
        pct = 100 * counts / total if total else counts
        bars = ax.bar(['Comprar POL', 'Vender POL', 'Mantener'], pct, color=colors)
        for bar, n, pc in zip(bars, counts, pct):
            ax.annotate(f'{n} ({pc:.1f}%)', (bar.get_x()+bar.get_width()/2, pc),
                        xytext=(0, 4), textcoords='offset points', ha='center', fontsize=9)
        ax.set(title=f'{etapa}: {total} decisiones', ylabel='Porcentaje de decisiones (%)', ylim=(0, 115))
        ax.grid(axis='y', alpha=.2)
        ax.set_axisbelow(True)
    fig_acc.tight_layout()

    fig_est, axes = plt.subplots(1, 2, figsize=(10.5, 4))
    for ax, etapa in zip(axes, ('Train', 'Test')):
        mat = matriz_transiciones_regimen(analisis[etapa.lower()])
        # Escalas independientes: train y test tienen volúmenes muy diferentes.
        image = ax.imshow(mat.to_numpy(), cmap='Blues', vmin=0,
                          vmax=max(1, mat.to_numpy().max()))
        ax.set_xticks(range(len(mat.columns)), mat.columns)
        ax.set_yticks(range(len(mat.index)), mat.index)
        ax.set(xlabel='Régimen siguiente', ylabel='Régimen actual',
               title=f'{etapa}: {int(mat.to_numpy().sum())} transiciones')
        for i in range(len(mat.index)):
            for j in range(len(mat.columns)):
                v = int(mat.iloc[i, j])
                ax.text(j, i, str(v), ha='center', va='center',
                        color='white' if v > mat.to_numpy().max()/2 else '#17324d')
    fig_est.tight_layout()
    return fig_acc, fig_est
