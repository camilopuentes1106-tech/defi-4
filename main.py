"""
main.py — Configuración y ejecución del pipeline POL/USDC (Uniswap V3, Polygon).
"""

from swap import SwapPipeline
from wallet import WalletView

# ──────────────────────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────────────────────

RPC_URL      = "https://polygon-mainnet.g.alchemy.com/v2/<TU_API_KEY>"
POOL_ADDRESS = "0xA374094527e1673A86dE625aa59517c5dE346d32"  # WPOL/USDC 0.05%

TICKER_YF    = "POL-USD"
DECIMALES_T0 = 18   # WPOL
DECIMALES_T1 = 6    # USDC
NOMBRE_T0    = "pol"
NOMBRE_T1    = "usdc"

HORAS        = 24
HORIZONTES   = ["1m", "5m", "15m", "1h"]
CACHE_PATH   = "cache/swaps_pol_usdc_24h.parquet"

# ──────────────────────────────────────────────────────────────────────────────
# PIPELINE
# ──────────────────────────────────────────────────────────────────────────────

pipeline = SwapPipeline(
    rpc_url      = RPC_URL,
    pool_address = POOL_ADDRESS,
    ticker_yf    = TICKER_YF,
    decimales_t0 = DECIMALES_T0,
    decimales_t1 = DECIMALES_T1,
    nombre_t0    = NOMBRE_T0,
    nombre_t1    = NOMBRE_T1,
    cache_path   = CACHE_PATH,
)

swaps = pipeline.ejecutar(
    horas            = HORAS,
    horizontes       = HORIZONTES,
    forzar_descarga  = False,
    verbose          = True,
)

print("\n── TABLA SWAPS ──")
print(swaps.head().to_string())

# ──────────────────────────────────────────────────────────────────────────────
# VISTA WALLETS
# ──────────────────────────────────────────────────────────────────────────────

wallets = WalletView(pipeline, horizontes=HORIZONTES)
wallets.construir()

print("\n── VISTA WALLETS ──")
print(wallets.df.head(20).to_string())

print("\n── TOP 10 BUYERS (1h) ──")
print(wallets.top_wallets(horizonte="1h", n=10, direccion="Buy").to_string())
