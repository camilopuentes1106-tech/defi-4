# Proyecto DEFI IV: agente RL para operar POL guiado por wallets on-chain

## 1. ¿Qué problema resolvemos?

Queremos que un agente decida cada hora si debe comprar POL, vender POL o
mantener su posición. El agente no mira solamente el precio: también usa el
comportamiento pasado de wallets que han mostrado resultados consistentes.

Ejemplo: si varias wallets que históricamente han acertado en compras de POL a
una hora están comprando ahora, esto puede ser una señal favorable. Aun así, el
agente no las copia de forma automática: compara esa señal con el retorno y el
riesgo estimados.

El objetivo no es predecir cada movimiento del precio. El objetivo es aprender
una política que maximice la riqueza acumulada neta de gas.

---

## 2. Datos de entrada: swaps POL/USDC

Cada fila original representa un evento de swap del pool Uniswap V3 POL/USDC.
Una transacción puede tener varios eventos si pasó por varias rutas del DEX.
Por eso se convierte primero en un **swap lógico** por `wallet × transacción`.

Ejemplo de ruta multihop:

```text
Wallet A intercambia USDC → WETH → POL dentro de la transacción 0xabc.
```

Aunque haya dos eventos técnicos, para el análisis es una sola decisión de la
wallet: compró POL. Se agrupan los eventos, se calcula el POL neto y el gas se
asigna una sola vez; de este modo no se duplica el costo.

---

## 3. Acciones observadas en las wallets

De los swaps lógicos salen tres tipos de acción:

| Acción | Significado | Ejemplo |
|---|---|---|
| `BUY_POL` | La wallet aumenta su cantidad de POL. | Paga 100 USDC y recibe 50 POL. |
| `SELL_POL` | La wallet reduce su cantidad de POL. | Entrega 50 POL y recibe 105 USDC. |
| `HOLD` | Conserva un lote comprado durante un período. | Compra 50 POL a las 10:00 y los vende a las 14:00. |

`HOLD` no significa “no hubo swap”. Se reconstruye con FIFO: cada venta se
empareja con las compras más antiguas disponibles. Sólo se considera Hold si se
observó tanto la compra como el cierre posterior.

Ejemplo FIFO:

```text
10:00  compra 10 POL a 1.00 USDC
11:00  compra 10 POL a 1.10 USDC
12:00  vende 15 POL a 1.20 USDC
```

La venta cierra primero los 10 POL de las 10:00 y luego 5 POL de las 11:00. Se
obtienen dos ciclos Hold con distinta duración y distinto PnL.

---

## 4. Decisión madura, pendiente y sin fuga de información

Una decisión de horizonte 1 h necesita conocer el precio una hora después para
saber si fue buena o mala. Antes de esa hora la decisión está `pending`.

Ejemplo:

```text
10:00  Wallet A compra POL.
10:30  Corte actual del agente: la compra sigue PENDING.
11:00  Ya existe precio posterior: la compra queda madura y puede evaluarse.
```

En el corte de las 10:30 no se permite usar el resultado que se conocerá a las
11:00. Esta regla evita **fuga temporal** o *data leakage*: usar en el pasado
un dato que en realidad sólo se conoció después.

---

## 5. Métricas de una wallet

Cada perfil se calcula para una combinación:

```text
wallet × acción × horizonte
```

Por ejemplo, una misma wallet puede ser buena comprando POL a 1 h, pero no
necesariamente vendiendo POL o manteniendo a 15 minutos.

### n_decisiones

Número de decisiones maduras usadas para medir el perfil.

Ejemplo: si la wallet hizo 6 compras a 1 h, pero las dos últimas aún no llegan
a su precio forward, entonces `n_decisiones = 4` y `n_pendientes = 2`.

### pnl_neto_usdc

Suma de la ganancia o pérdida en USDC, descontando gas para Buy y Sell.

Ejemplo: ganancias de `+3`, `+2` y pérdida de `-1` USDC dan
`pnl_neto_usdc = 4` USDC.

### retorno_neto_mediano

Mediana de retornos netos. La mediana es menos sensible a un caso extremo que
el promedio.

Ejemplo: retornos `[1 %, 2 %, 3 %, 30 %]`. El promedio es 9 %, pero la mediana
es 2.5 %; la mediana representa mejor el resultado típico.

### tasa_acierto

Proporción de decisiones maduras con retorno neto positivo.

Ejemplo: 8 decisiones, de las cuales 6 son positivas:

```text
tasa_acierto = 6 / 8 = 0.75 = 75 %
```

### profit_factor

Ganancias brutas divididas entre pérdidas brutas. Se limita a 3 para que una
sola racha sin pérdidas no domine todo el score.

Ejemplo: ganancias acumuladas 12 USDC y pérdidas acumuladas 4 USDC:

```text
profit_factor = 12 / 4 = 3
```

Si el valor calculado fuese 5, el modelo usa 3 en el score.

---

## 6. Consistency score y wallets ganadoras

El score de consistencia combina precisión, relación ganancia/pérdida y
cantidad de evidencia:

```text
consistency_score = 0.50 × tasa_acierto
                  + 0.30 × (min(profit_factor, 3) / 3)
                  + 0.20 × min(n_decisiones / 10, 1)
```

Ejemplo 1: una wallet con 8 decisiones, 75 % de acierto y profit factor 2.4.

```text
0.50 × 0.75 + 0.30 × (2.4 / 3) + 0.20 × (8 / 10)
= 0.375 + 0.240 + 0.160
= 0.775
```

Su score es 0.775. Es un perfil interesante, pero no alcanza el umbral RL de
0.80.

Ejemplo 2: 10 decisiones, 80 % de acierto y profit factor 3.

```text
0.50 × 0.80 + 0.30 × 1 + 0.20 × 1 = 0.90
```

Para ser `winner` el score por sí solo no basta. Además se exige mínimo tres
decisiones maduras, PnL neto acumulado positivo y retorno neto mediano positivo.
Para dirigir al agente se aplica el filtro más exigente:

```text
horizonte = 1h
winner_status = winner
consistency_score >= 0.80
```

---

## 7. De muchas wallets a una señal única

En cada hora se suman los scores de las wallets dirigidas según su acción.

Ejemplo:

| Dirección | Suma de scores |
|---|---:|
| Buy | 2.70 |
| Sell | 0.80 |
| Hold | 1.10 |

La señal es `BUY`, porque Buy tiene el soporte mayor. Su confianza es:

```text
(2.70 - 1.10) / (2.70 + 0.80 + 1.10) = 0.348
```

Si Buy y Sell tuvieran exactamente el mismo soporte máximo, la señal sería
`NEUTRAL`. Esto evita inventar una preferencia cuando las wallets no están de
acuerdo.

---

## 8. Formulación del MDP

El problema se representa con un Proceso de Decisión de Markov:

```text
M = (S, A, P, R, gamma)
```

| Símbolo | En este proyecto |
|---|---|
| `S` | Estados del mercado, señal de wallets y posición. |
| `A` | Buy, Sell o Hold cuando sea válido. |
| `P` | Transición empírica entre estados. |
| `R` | Recompensa financiera con gas y bonus prudente. |
| `gamma` | Descuento de valor futuro; usamos 0.99. |

El agente toma una decisión cada hora. El episodio inicial contiene 24 pasos,
equivalentes a una ventana de 24 horas.

---

## 9. Estado del agente

El estado es:

```text
estado = (regimen_mercado, senal_wallets, posicion)
```

### Régimen de mercado

Se usan retornos pasados de 1m, 5m, 15m y 1h. La media de esos retornos se
clasifica así:

| Régimen | Regla con banda ±0.10 % | Ejemplo |
|---|---|---|
| `UP` | impulso mayor a +0.10 % | Retorno compuesto de +0.35 %. |
| `FLAT` | entre -0.10 % y +0.10 % | Retorno compuesto de +0.03 %. |
| `DOWN` | impulso menor a -0.10 % | Retorno compuesto de -0.42 %. |

La banda es fija. No se calculan cuantiles usando horas futuras, porque eso
haría que el estado histórico conociera información que no tenía en ese momento.

### Señal de wallets

Puede ser `BUY`, `SELL`, `HOLD` o `NEUTRAL`.

### Posición

`0` significa que el capital está en USDC y `1` significa que está invertido en
POL. Hay `3 × 4 × 2 = 24` estados interpretables.

Ejemplo de estado completo:

```text
(UP, BUY, 0)
```

Significa: el mercado tiene impulso alcista, las wallets favorecen compras y el
agente todavía está en USDC.

---

## 10. Acciones admisibles

El modelo no permite posiciones cortas. Por eso las acciones dependen de la
posición actual.

| Posición actual | Acciones posibles | Ejemplo |
|---|---|---|
| `0 = USDC` | `BUY_POL`, `HOLD` | Puede comprar o esperar con USDC. |
| `1 = POL` | `SELL_POL`, `HOLD` | Puede vender POL o conservarlo. |

Ejemplo: en el estado `(DOWN, SELL, 1)`, vender POL es válido. En cambio,
`BUY_POL` no se ofrece porque el modelo ya está completamente invertido en POL.

---

## 11. Transiciones empíricas

Una transición describe cómo puede cambiar el estado económico de una hora a
la siguiente. Se cuentan cambios realmente vistos en los datos y se aplica
suavizado de Laplace con `alpha = 1`.

Ejemplo: desde el estado económico `(UP, BUY)` se observaron cuatro cambios:

```text
3 veces hacia (UP, BUY)
1 vez hacia (DOWN, SELL)
```

Existen 12 estados económicos posibles (`3 regímenes × 4 señales`). Después de
Laplace, el denominador es `4 observaciones + 12 = 16`.

```text
P(UP, BUY)     = (3 + 1) / 16 = 0.25
P(DOWN, SELL)  = (1 + 1) / 16 = 0.125
P(cada estado no visto) = 1 / 16 = 0.0625
```

La acción fija la posición siguiente. Si el agente estaba en USDC y compra,
todos los estados siguientes posibles tendrán posición POL.

---

## 12. Recompensa del agente

Primero se calcula el retorno de POL durante la hora:

```text
retorno_POL = precio_t+1 / precio_t - 1
```

La recompensa base depende de la posición que queda tras la acción y del gas:

| Situación | Acción | Recompensa base |
|---|---|---|
| Está en USDC | `BUY_POL` | `retorno_POL - gas` |
| Está en USDC | `HOLD` | `0` |
| Está en POL | `SELL_POL` | `-gas` |
| Está en POL | `HOLD` | `retorno_POL` |

El precio de la hora siguiente se usa únicamente para entrenar y evaluar sobre
el histórico. En vivo, el agente sólo conoce el estado y la política aprendida
hasta la hora actual.

### Bonus de wallets

La señal de wallets sólo refuerza una acción cuando coincide con una ventaja
financiera realizada:

```text
R_final = R_base + 0.25 × confianza_wallets × ventaja_realizada
```

Ejemplo: desde USDC, POL sube de 1.00 a 1.02; gas = 0.001 y la confianza Buy =
0.80.

```text
R_base(BUY) = 0.02 - 0.001 = 0.019
R_base(HOLD) = 0
ventaja(BUY) = 0.019 - 0 = 0.019
bonus = 0.25 × 0.80 × 0.019 = 0.0038
R_final(BUY) = 0.0228
```

El bonus aumenta una decisión que ya era buena. No transforma una compra mala
en buena: si POL cae, el retorno y la ventaja de Buy son negativos.

---

## 13. Ecuación de Bellman

Bellman compara la recompensa inmediata con el valor esperado de las horas
siguientes:

```text
V_t(s) = max_a [ R(s,a) + gamma × suma(P(s'|s,a) × V_t+1(s')) ]
```

Ejemplo simple: en un estado, Buy tiene recompensa inmediata 0.019 y un valor
futuro esperado de 0.010. Hold tiene recompensa inmediata 0 y valor futuro
esperado 0.015.

```text
Q(Buy)  = 0.019 + 0.99 × 0.010 = 0.0289
Q(Hold) = 0     + 0.99 × 0.015 = 0.01485
```

La política elige Buy porque 0.0289 es mayor. Bellman se resuelve hacia atrás
en las 24 horas del episodio para obtener una acción recomendada para cada
estado.

---

## 14. Replay histórico y simulación

El **replay histórico** toma precios que ya ocurrieron y aplica la política
hora por hora. Sirve para explicar qué habría hecho el agente.

Ejemplo:

| Hora | Estado | Acción de la política | Recompensa | Riqueza |
|---|---|---|---:|---:|
| 10:00 | `(UP, BUY, 0)` | `BUY_POL` | 0.0228 | 102.28 |
| 11:00 | `(FLAT, HOLD, 1)` | `HOLD` | 0.0040 | 102.69 |
| 12:00 | `(DOWN, SELL, 1)` | `SELL_POL` | -0.0010 | 102.59 |

Si la riqueza inicial era 100 USDC, la tabla permite explicar cómo evolucionó.
No prueba que el modelo será rentable en el futuro, porque está usando la misma
ventana con la cual se estimó el modelo.

La **simulación MDP** es diferente: en vez de seguir exactamente el precio
histórico, toma transiciones al azar con las probabilidades empíricas estimadas.
Sirve para ilustrar escenarios posibles dentro del modelo.

---

## 15. Cómo leer las gráficas

### Señal causal de wallets por hora

- Eje X: hora de decisión.
- Eje Y: confianza de la señal dominante.
- Color: `BUY`, `SELL`, `HOLD` o `NEUTRAL`.

Ejemplo: un punto `BUY` con confianza 0.35 indica que las wallets Buy tenían
más soporte que Sell y Hold, pero no que haya certeza de 35 % de ganancia.

### Política Bellman por estado

Muestra la acción recomendada para cada combinación de régimen, señal y
posición.

Ejemplo: si para `(DOWN, SELL, 1)` aparece `SELL_POL`, el modelo considera que
conservar POL tiene menor valor esperado que venderlo en ese estado.

### Replay de riqueza

Muestra la riqueza acumulada desde una base de 100.

Ejemplo: pasar de 100 a 103 significa un incremento acumulado de 3 % durante
el replay, no una promesa de rendimiento futuro.

---

## 16. Resultado actual y límites

En el snapshot de ejemplo se encontraron 82 perfiles `wallet × acción × 1h`
que pasan el filtro de 0.80, correspondientes a 66 wallets únicas. Las señales
horarias observadas fueron principalmente Hold. Estos números cambian cuando se
use otro snapshot o se acumulen más ventanas.

Las limitaciones principales son:

1. **Poca historia:** 24 horas no bastan para afirmar que una política es
   estable. Se deben acumular snapshots de muchas fechas.
2. **Evaluación in-sample:** el replay actual explica la muestra de
   entrenamiento. Debe existir una prueba posterior fuera de muestra.
3. **Modelo simplificado:** sólo representa USDC/POL long-only, no balances
   externos, transferencias, posiciones cortas, slippage ni impacto de mercado.
4. **Wallets no son oráculos:** una wallet consistente puede cambiar de
   estrategia. El score es una evidencia estadística, no una garantía.

El siguiente paso metodológico es separar los datos por tiempo: entrenar con
snapshots antiguos, ajustar parámetros en una validación posterior y medir el
resultado final sobre horas nunca usadas para construir perfiles o transiciones.
