"""Permite ejecutar la librería con ``python -m defi4``."""

from .pipeline.wallets import main


if __name__ == "__main__":
    raise SystemExit(main())
