from __future__ import annotations

import logging
import os
from datetime import datetime


def build_logger(log_dir: str) -> logging.Logger:
    """Construeix i configura el sistema de registre de traces (logger) de l'aplicació.

    Crea un logger configurat per escriure simultàniament en un fitxer de text amb una
    marca de temps (timestamp) dins del directori ``log_dir`` i a la consola de sortida.
    El gestor del fitxer (file handler) captura traces a partir del nivell DEBUG, mentre
    que el gestor de la consola (console handler) captura a partir del nivell INFO.

    Parameters
    ----------
    log_dir : str
        Ruta del directori on es crearà i s'emmagatzemarà el fitxer de log.

    Returns
    -------
    logging.Logger
        Instància del logger completament configurada amb el nom ``"recommender_system"``.
    """
    os.makedirs(log_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_path = os.path.join(log_dir, f"log_{timestamp}.txt")

    logger = logging.getLogger("recommender_system")
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    if logger.handlers:
        logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s | %(levelname)-8s | %(name)s | %(message)s")

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    logger.debug("Logger inicialitzat. Fitxer de log: %s", log_path)
    return logger
