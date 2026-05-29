import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Union


def setup_logging(base_dir: Union[str, Path] = ".") -> None:
    log_dir = Path(base_dir) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    root.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    system_handler = RotatingFileHandler(
        log_dir / "system.log",
        maxBytes=2_000_000,
        backupCount=5,
        encoding="utf-8",
    )
    system_handler.setLevel(logging.DEBUG)
    system_handler.setFormatter(formatter)

    error_handler = RotatingFileHandler(
        log_dir / "errors.log",
        maxBytes=1_000_000,
        backupCount=5,
        encoding="utf-8",
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(formatter)

    root.addHandler(system_handler)
    root.addHandler(error_handler)

    performance_logger = logging.getLogger("performance")
    performance_logger.setLevel(logging.INFO)
    performance_logger.propagate = False
    performance_logger.handlers.clear()

    performance_handler = RotatingFileHandler(
        log_dir / "performance.log",
        maxBytes=1_000_000,
        backupCount=5,
        encoding="utf-8",
    )
    performance_handler.setFormatter(formatter)
    performance_logger.addHandler(performance_handler)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
