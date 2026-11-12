"""Crear gráficos y cifras de las decisiones del mismo experimento DQN."""
import json
from pathlib import Path
import matplotlib.pyplot as plt
from lib_rl.explicacion_dqn import analizar_decisiones_dqn, graficar_decisiones_dqn, matriz_transiciones_regimen


def guardar_decisiones(experimento, folder):
    folder = Path(folder)
    out = folder / 'figuras'
    out.mkdir(exist_ok=True)
    a = analizar_decisiones_dqn(experimento)
    figs = graficar_decisiones_dqn(a)
    for fig, name in zip(figs, ('dqn_acciones_train_test.pdf', 'dqn_transiciones_train_test.pdf')):
        fig.savefig(out/name, bbox_inches='tight')
        plt.close(fig)
    primero = a['test'].iloc[0]
    def estado(s):
        return '/'.join([str(s[0]), str(s[1]), 'POL' if s[2] else 'USDC'])
    commands = {
        'DQNTrainDecisiones': len(a['train']), 'DQNTestDecisiones': len(a['test']),
        'DQNTrainExploracion': f"{100*a['train'].exploracion.mean():.1f}",
        'DQNTrainTransiciones': int(matriz_transiciones_regimen(a['train']).to_numpy().sum()),
        'DQNTestTransiciones': int(matriz_transiciones_regimen(a['test']).to_numpy().sum()),
        'DQNPrimerEstado': estado(primero.estado),
        'DQNPrimerSiguiente': estado(primero.estado_siguiente) if primero.estado_siguiente else 'Final',
        'DQNPrimerAccion': primero.accion.replace('_', r'\_'),
        'DQNPrimerQ': f"{primero.Q_accion:.1f}",
        'DQNPrimerQAlternativa': f"{primero.Q_alternativa:.1f}",
    }
    (folder/'resultados_decisiones.tex').write_text(
        '% Generado con el registro del experimento DQN.\n' + ''.join(
            '\\newcommand{\\'+k+'}{'+str(v)+'}\n' for k,v in commands.items()), encoding='utf8')
    info = {'acciones': {col: {str(k): int(v) for k,v in a['acciones'][col].items()} for col in a['acciones']},
            'exploracion_pct': float(100*a['train'].exploracion.mean()),
            'transiciones_train': int(matriz_transiciones_regimen(a['train']).to_numpy().sum()),
            'transiciones_test': int(matriz_transiciones_regimen(a['test']).to_numpy().sum()),
            'primero_estado': estado(primero.estado), 'primero_siguiente': estado(primero.estado_siguiente),
            'primero_accion': primero.accion, 'primero_q': float(primero.Q_accion),
            'primero_alternativa': float(primero.Q_alternativa)}
    (out/'dqn_decisiones.json').write_text(json.dumps(info,ensure_ascii=False,indent=2), encoding='utf8')
    return info
