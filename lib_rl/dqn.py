"""DQN para el MDP long-only de POL; datos históricos, no trading real."""
from collections import deque
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F

from .bellman import (ACCIONES, REGIMENES_MERCADO, SENALES_WALLETS,
                      acciones_admisibles, posicion_despues,
                      calcular_recompensa_con_perfiles, _support_from_row)
from .qlearning import QLearningResult
from .bellman import construir_mdp_empirico, resolver_bellman_finito


def vector_estado(row, posicion, t, horizonte):
    """Solo información disponible al decidir: categorías, posición y etapa."""
    return np.array([*[float(row.regimen_mercado == v) for v in REGIMENES_MERCADO],
                     *[float(row.senal_wallets == v) for v in SENALES_WALLETS],
                     float(posicion), t / max(horizonte - 1, 1)], dtype=np.float32)


def indices_validos(posicion):
    return [ACCIONES.index(a) for a in acciones_admisibles(posicion)]


class RedQ(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(9, 32), nn.ReLU(), nn.Linear(32, 32),
                                 nn.ReLU(), nn.Linear(32, 3))

    def forward(self, x):
        return self.net(x)


@dataclass
class ResultadoDQN:
    red: RedQ
    objetivo: RedQ
    memoria: deque
    registro: pd.DataFrame
    perdidas: list
    actualizaciones: int
    sincronizaciones: int
    horizonte: int
    profile_weight: float


def entrenar_dqn(bloques, *, episodios=600, gamma=.99, epsilon_final=.05,
                 profile_weight=.25, seed=7, batch=64, target_cada=120):
    if not bloques or len({len(b) for b in bloques}) != 1:
        raise ValueError('Se requieren bloques no vacíos de igual duración.')
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    rng = np.random.default_rng(seed)
    net, target = RedQ(), RedQ()
    target.load_state_dict(net.state_dict())
    optim = torch.optim.Adam(net.parameters(), lr=.001)
    memoria = deque(maxlen=8000)
    log, losses = [], []
    updates = syncs = 0
    H = len(bloques[0])
    for ep in range(episodios):
        frame = bloques[int(rng.integers(len(bloques)))].reset_index(drop=True)
        eps = .9 + (epsilon_final - .9) * ep / max(episodios - 1, 1)
        pos, total = 0, 0.
        for t, row in frame.iterrows():
            x = vector_estado(row, pos, t, H)
            valid = indices_validos(pos)
            with torch.no_grad():
                q = net(torch.from_numpy(x)).numpy()
            a = int(rng.choice(valid)) if rng.random() < eps else valid[int(np.argmax(q[valid]))]
            action = ACCIONES[a]
            reward = calcular_recompensa_con_perfiles(posicion=pos, accion=action,
                precio_t=float(row.precio_t), precio_t1=float(row.precio_t1),
                costo_gas_ratio=float(row.costo_gas_ratio), support_wallets=_support_from_row(row),
                profile_weight=profile_weight)['reward_final'] * 1e4
            next_pos = posicion_despues(pos, action)
            done = t == H - 1
            nx = np.zeros(9, np.float32) if done else vector_estado(frame.iloc[t+1], next_pos, t+1, H)
            memoria.append((x, a, reward, nx, float(done), next_pos))
            pos, total = next_pos, total + reward
            if len(memoria) >= batch:
                batch_rows = [memoria[int(i)] for i in rng.integers(len(memoria), size=batch)]
                xs = torch.tensor(np.stack([z[0] for z in batch_rows]))
                aa = torch.tensor([z[1] for z in batch_rows], dtype=torch.long)
                rr = torch.tensor([z[2] for z in batch_rows], dtype=torch.float32)
                nx = torch.tensor(np.stack([z[3] for z in batch_rows]))
                done_b = torch.tensor([z[4] for z in batch_rows], dtype=torch.float32)
                mask = torch.zeros((batch, len(ACCIONES)), dtype=torch.bool)
                for i, z in enumerate(batch_rows):
                    mask[i, indices_validos(z[5])] = True
                pred = net(xs).gather(1, aa[:, None]).squeeze(1)
                with torch.no_grad():
                    future = target(nx).masked_fill(~mask, -torch.inf).max(1).values
                    y = rr + gamma * (1 - done_b) * future
                loss = F.smooth_l1_loss(pred, y)
                optim.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 5.)
                optim.step(); updates += 1; losses.append(float(loss.item()))
                if updates % target_cada == 0:
                    target.load_state_dict(net.state_dict()); syncs += 1
        log.append({'episodio': ep+1, 'epsilon': eps, 'recompensa_pb': total,
                    'memoria': len(memoria)})
    net.eval(); target.eval()
    return ResultadoDQN(net, target, memoria, pd.DataFrame(log), losses, updates, syncs, H, profile_weight)


def auditar(modelo, gamma=.99):
    x, a, r, nx, done, pos = next(z for z in modelo.memoria if not z[4])
    with torch.no_grad():
        q = modelo.red(torch.from_numpy(x)).numpy()
        futuro = modelo.objetivo(torch.from_numpy(nx)).numpy()
    valid = indices_validos(pos)
    y = r + gamma * (1-done) * float(np.max(futuro[valid]))
    loss = F.smooth_l1_loss(torch.tensor(float(q[a])), torch.tensor(float(y))).item()
    return {'estado': x, 'accion': ACCIONES[a], 'recompensa_pb': r, 'estado_siguiente': nx,
            'done': done, 'Q_actual': q, 'Q_accion': float(q[a]), 'Q_objetivo': futuro,
            'acciones_futuras_validas': [ACCIONES[i] for i in valid], 'target_pb': y, 'huber': loss}


def entrenar_tabla_bloques(bloques, mdp, *, episodes=2000, gamma=.99, seed=7):
    """Q tabular sobre los mismos bloques y recompensa del DQN."""
    rng = np.random.default_rng(seed)
    h = len(bloques[0]); q = np.zeros((h, len(mdp.states), len(ACCIONES)))
    visits = np.zeros_like(q, dtype=int); log = []
    for ep in range(episodes):
        frame = bloques[int(rng.integers(len(bloques)))].reset_index(drop=True)
        eps = .9 + (.05 - .9) * ep / max(episodes-1, 1)
        pos = 0; total = 0.
        for t, row in frame.iterrows():
            state = mdp.states.index((row.regimen_mercado, row.senal_wallets, pos))
            ids = indices_validos(pos)
            a = int(rng.choice(ids)) if rng.random() < eps else ids[int(np.argmax(q[t,state,ids]))]
            action = ACCIONES[a]
            reward = calcular_recompensa_con_perfiles(posicion=pos, accion=action,
                precio_t=float(row.precio_t), precio_t1=float(row.precio_t1),
                costo_gas_ratio=float(row.costo_gas_ratio), support_wallets=_support_from_row(row),
                profile_weight=mdp.profile_weight)['reward_final']
            next_pos = posicion_despues(pos, action)
            future = 0.
            if t < h-1:
                nxt = frame.iloc[t+1]
                ns = mdp.states.index((nxt.regimen_mercado,nxt.senal_wallets,next_pos))
                future = np.max(q[t+1,ns,indices_validos(next_pos)])
            q[t,state,a] += .15 * (reward + gamma * future - q[t,state,a])
            visits[t,state,a] += 1; total += reward; pos = next_pos
        log.append({'episode':ep+1, 'epsilon':eps, 'recompensa_total':total, 'q_abs_media':np.abs(q).mean()})
    return QLearningResult(q, visits, pd.DataFrame(log), .9, .05, .15, gamma)


def evaluar_bloque(frame, metodo, modelo, mdp, *, profile_weight=.25):
    """Evalúa política congelada; riqueza económica excluye bonus de perfil."""
    frame = frame.reset_index(drop=True)
    pos, wealth, objetivo, rows = 0, 100., 100., []
    for t, row in frame.iterrows():
        q = None
        allowed = acciones_admisibles(pos)
        if metodo == 'DQN':
            with torch.no_grad():
                q = modelo.red(torch.from_numpy(vector_estado(row, pos, t, len(frame)))).numpy()
            ids = indices_validos(pos)
            action = ACCIONES[ids[int(np.argmax(q[ids]))]]
        elif metodo == 'Q-learning':
            state = (row.regimen_mercado, row.senal_wallets, pos)
            ids = indices_validos(pos)
            q = modelo.q_values[t, mdp.states.index(state)]
            action = ACCIONES[ids[int(np.argmax(q[ids]))]]
        elif metodo == 'Bellman':
            state = (row.regimen_mercado, row.senal_wallets, pos)
            action = str(modelo.policy[t, mdp.states.index(state)])
        elif metodo == 'Buy and Hold':
            action = 'BUY_POL' if pos == 0 else 'HOLD'
        else:
            raise ValueError('Método desconocido')
        reward = calcular_recompensa_con_perfiles(posicion=pos, accion=action,
            precio_t=float(row.precio_t), precio_t1=float(row.precio_t1),
            costo_gas_ratio=float(row.costo_gas_ratio), support_wallets=_support_from_row(row),
            profile_weight=profile_weight)
        new_pos = posicion_despues(pos, action)
        wealth *= 1 + reward['reward_base']; objetivo *= 1 + reward['reward_final']
        rows.append({'as_of':row.as_of, 'metodo':metodo, 't':t, 'estado':str((row.regimen_mercado,row.senal_wallets,pos)),
                     'accion':action, 'posicion':new_pos, 'reward_base':reward['reward_base'],
                     'reward_final':reward['reward_final'], 'wealth':wealth, 'wealth_objetivo':objetivo,
                     'Q_BUY_POL':float(q[0]) if metodo == 'DQN' else np.nan,
                     'Q_SELL_POL':float(q[1]) if metodo == 'DQN' else np.nan,
                     'Q_HOLD':float(q[2]) if metodo == 'DQN' else np.nan})
        pos = new_pos
    return pd.DataFrame(rows)


def experimento_pol(observaciones, swaps, *, horizonte=4, episodios=600, seed=7):
    """Split cronológico; selección en validación y evaluación congelada en test."""
    obs = observaciones.sort_values('as_of').reset_index(drop=True).copy()
    ultima_fecha = pd.to_datetime(swaps['timestamp'], utc=True).max()
    # No dar por observada una hora que termina después del último dato.
    obs = obs[obs.as_of + pd.Timedelta(hours=1) <= ultima_fecha].reset_index(drop=True)
    # Los bloques completos se construyen antes de separar; no reutilizan horas.
    blocks = [obs.iloc[i:i+horizonte].copy().reset_index(drop=True)
              for i in range(0, len(obs)-horizonte+1, horizonte)]
    if len(blocks) < 3:
        raise ValueError('Se necesitan al menos tres bloques completos para train, validación y test.')
    for b in blocks:
        if not b.as_of.diff().dropna().eq(pd.Timedelta(hours=1)).all():
            raise ValueError('La trayectoria contiene horas faltantes; se requieren bloques contiguos.')
    train, valid, test = blocks[:-2], [blocks[-2]], [blocks[-1]]
    # La mediana de gas se estima solo con swaps anteriores al primer corte de validación.
    sw = swaps.copy()
    sw['timestamp'] = pd.to_datetime(sw['timestamp'], utc=True)
    sw = sw[sw.timestamp < valid[0].as_of.iloc[0]]
    col = 'usdc_cantidad' if 'usdc_cantidad' in sw else 'notional_usdc'
    ratios = (pd.to_numeric(sw['gas_usdc'], errors='coerce') /
              pd.to_numeric(sw[col], errors='coerce')).replace([np.inf,-np.inf], np.nan).dropna()
    ratios = ratios[ratios >= 0]
    if ratios.empty:
        raise ValueError('No hay datos de gas válidos anteriores a validación.')
    gas = float(ratios.median())
    for b in train + valid + test:
        b['costo_gas_ratio'] = gas
    mdp = construir_mdp_empirico(pd.concat(train, ignore_index=True), profile_weight=.25)
    tab = entrenar_tabla_bloques(train, mdp, seed=seed)
    bellman = resolver_bellman_finito(mdp, horizon=horizonte, gamma=.99)
    candidatos = {eps: entrenar_dqn(train, episodios=episodios, epsilon_final=eps, seed=seed)
                  for eps in (.05, .30)}
    validacion = []
    for eps, model in candidatos.items():
        replay = evaluar_bloque(valid[0], 'DQN', model, mdp)
        validacion.append({'epsilon_final':eps, 'retorno_economico':replay.wealth.iloc[-1]/100-1})
    validacion = pd.DataFrame(validacion)
    ganador = float(validacion.sort_values(['retorno_economico','epsilon_final'], ascending=[False, True]).iloc[0].epsilon_final)
    modelos = {'DQN':candidatos[ganador], 'Q-learning':tab, 'Bellman':bellman, 'Buy and Hold':None}
    resultados = pd.concat([evaluar_bloque(test[0], name, model, mdp)
                            for name, model in modelos.items()], ignore_index=True)
    escenario = test[0].copy()
    escenario['senal_wallets'] = 'NEUTRAL'
    for col in ('support_buy','support_sell','support_hold','confianza_wallets'):
        escenario[col] = 0.
    sin_wallets = pd.concat([evaluar_bloque(escenario, name, model, mdp)
                             for name, model in modelos.items()], ignore_index=True)
    return {'train':train, 'validacion_bloques':valid, 'test':test, 'gas':gas, 'mdp':mdp,
            'validacion':validacion, 'epsilon_elegido':ganador, 'modelos':modelos,
            'resultados':resultados, 'sin_wallets':sin_wallets}


def resumen_resultados(replays):
    rows=[]
    for name, r in replays.groupby('metodo', sort=False):
        eq = np.r_[100., r.wealth.to_numpy()]
        rows.append({'Método':name, 'Retorno económico':eq[-1]/100-1,
                     'Máx. drawdown':np.min(eq/np.maximum.accumulate(eq)-1),
                     'Cambios':int(r.accion.isin(['BUY_POL','SELL_POL']).sum()),
                     'Objetivo acumulado':r.wealth_objetivo.iloc[-1]/100-1})
    return pd.DataFrame(rows)


def comparar_replay_historico(observaciones, swaps, *, max_horas=24,
                              episodios_dqn=600, episodios_tabla=2000,
                              profile_weight=.25, seed=7):
    """Replay común in-sample: los tres modelos se entrenan en el mismo período.

    Incluye solo horas cuyo final queda dentro de los datos observados. La última
    hora incompleta no se extrapola. Cada política comienza en USDC y mantiene
    su posición entre horas; la riqueza económica excluye el bonus de wallets.
    """
    if max_horas < 2:
        raise ValueError('El replay necesita al menos dos horas.')
    frame = observaciones.sort_values('as_of').copy().reset_index(drop=True)
    ultima_fecha = pd.to_datetime(swaps['timestamp'], utc=True).max()
    frame = frame[frame.as_of + pd.Timedelta(hours=1) <= ultima_fecha]
    frame = frame.iloc[:max_horas].copy().reset_index(drop=True)
    if len(frame) < 2:
        raise ValueError('No hay suficientes retornos horarios completos en el cache.')
    if not frame.as_of.diff().dropna().eq(pd.Timedelta(hours=1)).all():
        raise ValueError('El replay requiere horas consecutivas sin huecos.')

    mdp = construir_mdp_empirico(frame, profile_weight=profile_weight)
    q_tabular = entrenar_tabla_bloques([frame], mdp, episodes=episodios_tabla, seed=seed)
    dqn = entrenar_dqn([frame], episodios=episodios_dqn, profile_weight=profile_weight, seed=seed)
    bellman = resolver_bellman_finito(mdp, horizon=len(frame), gamma=.99)
    modelos = {'DQN': dqn, 'Q-learning': q_tabular, 'Bellman': bellman, 'Buy and Hold': None}
    replays = pd.concat([
        evaluar_bloque(frame, nombre, modelo, mdp, profile_weight=profile_weight)
        for nombre, modelo in modelos.items()
    ], ignore_index=True)
    return {'observaciones': frame, 'modelos': modelos, 'mdp': mdp,
            'resultados': replays, 'metricas': resumen_resultados(replays)}
