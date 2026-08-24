"""
wallet.py — WalletView
Agrega la tabla de swaps producida por SwapPipeline a nivel wallet/dirección,
calculando métricas de volumen y ganancia neta forward.
"""
from __future__ import annotations
import pandas as pd
from utils.swap import SwapPipeline

class WalletView:
    """
    Vista agregada por wallet y dirección de swap.

    Parámetros
    ----------
    pipeline : SwapPipeline
        Instancia ya ejecutada (pipeline.df debe estar poblado).
    horizontes : list[str] | None
        Horizontes forward a incluir. None toma todos los que existan en el df.
    """

    def __init__(
        self,
        pipeline: SwapPipeline,
        horizontes: list[str] | None = None,
    ) -> None:
        if pipeline.df.empty:
            raise ValueError(
                "SwapPipeline.df está vacío. "
                "Ejecutá pipeline.ejecutar() antes de construir WalletView."
            )

        self._pipeline   = pipeline
        self._horizontes = horizontes or self._detectar_horizontes(pipeline.df)
        self.df: pd.DataFrame = pd.DataFrame()

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _detectar_horizontes(df: pd.DataFrame) -> list[str]:
        """Infiere los horizontes presentes en el DataFrame."""
        horizontes = []
        for col in df.columns:
            if col.startswith("retorno_"):
                iv = col.replace("retorno_", "")
                horizontes.append(iv)
        return horizontes

    # ── API pública ───────────────────────────────────────────────────────────

    def construir(self) -> pd.DataFrame:
        """
        Construye la vista agregada y la guarda en self.df.

        Columnas resultantes
        --------------------
        wallet, direccion,
        n_swaps,
        {t0}_cantidad,   {t1}_cantidad,
        gas_{t1},        {t1}_neto,
        ganancia_neta_{t1}_{iv}   (por cada horizonte)
        """
        p   = self._pipeline
        df  = p.df.copy()

        col_t0      = p.col_cantidad_t0
        col_t1      = p.col_cantidad_t1
        col_gas_t1  = f"gas_{p.nombre_t1}"
        col_neto_t1 = f"{p.nombre_t1}_neto"

        # Compatibilidad: gas en t1 puede no existir si el df viene de cache
        if col_gas_t1 not in df.columns:
            df[col_gas_t1] = (
                df[f"gas_{p.nombre_t0}"] * df[p.col_precio]
            )
        if col_neto_t1 not in df.columns:
            df[col_neto_t1] = (
                df[col_t1] - df[col_gas_t1]
            ).clip(lower=0)

        agg: dict = {
            "n_swaps":      ("hash_tx",   "count"),
            col_t0:         (col_t0,      "sum"),
            col_t1:         (col_t1,      "sum"),
            col_gas_t1:     (col_gas_t1,  "sum"),
            col_neto_t1:    (col_neto_t1, "sum"),
        }

        for iv in self._horizontes:
            col_ganancia = f"ganancia_{p.nombre_t1}_{iv}"
            if col_ganancia in df.columns:
                agg[col_ganancia] = (col_ganancia, "sum")

        resultado = (
            df.groupby(["wallet", "direccion"])
            .agg(**agg)
            .reset_index()
        )

        # Ganancia neta = ganancia forward bruta − gas en t1
        for iv in self._horizontes:
            col_bruta = f"ganancia_{p.nombre_t1}_{iv}"
            col_neta  = f"ganancia_neta_{p.nombre_t1}_{iv}"

            if col_bruta in resultado.columns:
                resultado[col_neta] = (
                    resultado[col_bruta] - resultado[col_gas_t1]
                )
                resultado.drop(columns=[col_bruta], inplace=True)

        self.df = (
            resultado
            .sort_values(["wallet", "direccion"])
            .reset_index(drop=True)
        )

        return self.df

    def filtrar_wallet(self, address: str) -> pd.DataFrame:
        """Devuelve las filas de una wallet específica."""
        if self.df.empty:
            raise ValueError("Ejecutá construir() primero.")
        mask = self.df["wallet"].str.lower() == address.lower()
        return self.df[mask].reset_index(drop=True)

    def top_wallets(
        self,
        horizonte: str,
        n: int = 20,
        direccion: str | None = None,
    ) -> pd.DataFrame:
        """
        Devuelve las N wallets con mayor ganancia neta en un horizonte dado.

        Parámetros
        ----------
        horizonte : str
            Ej. "1h".
        n : int
            Cantidad de wallets a devolver.
        direccion : str | None
            "Buy", "Sell" o None para ambas.
        """
        if self.df.empty:
            raise ValueError("Ejecutá construir() primero.")

        col = f"ganancia_neta_{self._pipeline.nombre_t1}_{horizonte}"
        if col not in self.df.columns:
            raise ValueError(
                f"Columna '{col}' no encontrada. "
                f"Horizontes disponibles: {self._horizontes}"
            )

        df = self.df.copy()
        if direccion is not None:
            df = df[df["direccion"] == direccion]

        return (
            df.nlargest(n, col)
            .reset_index(drop=True)
        )

    def perfiles_por_horizonte(self, min_swaps: int = 3) -> pd.DataFrame:
        """Devuelve candidatas Buy/Sell para cada horizonte de ``WalletView``.

        La salida conserva una fila por wallet, posición y horizonte.  Una
        ``candidate_winner`` tiene al menos ``min_swaps`` y PnL neto agregado
        positivo. La confirmación como ``winner`` requiere el ledger detallado
        porque sólo allí se conocen los aciertos y pérdidas por operación.
        """
        if self.df.empty:
            raise ValueError("Ejecutá construir() primero.")
        from pol_wallet_summary import DEFAULT_HORIZONS, construir_perfiles_desde_tabla_wallets

        horizontes = tuple(h for h in self._horizontes if h in DEFAULT_HORIZONS)
        if not horizontes:
            horizontes = DEFAULT_HORIZONS

        return construir_perfiles_desde_tabla_wallets(
            self.df,
            horizontes=horizontes,
            min_swaps=min_swaps,
        )
