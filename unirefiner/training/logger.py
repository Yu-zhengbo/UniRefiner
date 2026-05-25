"""Training logging setup."""

from __future__ import annotations

import logging


def setup_logging(log_file: str | None, level: int = logging.INFO) -> None:
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d,%H:%M:%S",
    )
    logging.root.handlers.clear()
    logging.root.setLevel(level)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logging.root.addHandler(stream_handler)

    if log_file:
        file_handler = logging.FileHandler(filename=log_file)
        file_handler.setFormatter(formatter)
        logging.root.addHandler(file_handler)
