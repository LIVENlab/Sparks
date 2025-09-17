# logger_config.py
import logging
from pathlib import Path
import warnings
import sys

_TIME_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logging(
    log_dir: str = "logs",
    log_file: str = "sparks.log",
    verbosity: str = "INFO",
    capture_warnings: bool = True
) -> logging.Logger:

    Path(log_dir).mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("sparks")

    if logger.hasHandlers():
        logger.handlers.clear()
    
    file_handler = logging.FileHandler(f"{log_dir}/{log_file}", mode = "w")
    file_handler.setLevel(logging.DEBUG)

    file_formatter = logging.Formatter(
        "[%(asctime)s] %(name)s %(levelname)-8s %(message)s", datefmt=_TIME_FORMAT
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler(stream=sys.stdout)
    console_handler.setLevel(verbosity)  # user-controlled verbosity
    console_formatter = logging.Formatter(
        "%(levelname)-8s %(message)s"
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    # --- Capture warnings (optional) ---
    if capture_warnings:
        logging.captureWarnings(True)
        warnings_logger = logging.getLogger("py.warnings")
        warnings_logger.setLevel(logging.WARNING)
        warnings_logger.addHandler(console_handler)

    return logger



