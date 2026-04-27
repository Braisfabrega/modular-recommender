from __future__ import annotations

import logging


def build_logger(log_path: str) -> logging.Logger:
    logger = logging.getLogger("fase1_recommender")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if logger.handlers:
        logger.handlers.clear()

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger
