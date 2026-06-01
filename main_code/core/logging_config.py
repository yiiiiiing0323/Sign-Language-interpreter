import logging
import sys


def setup_logging(*_args, **_kwargs) -> None:
    """
    Configure console-only logging.
    The app no longer writes logs to disk.
    """
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    root.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(formatter)
    root.addHandler(console_handler)

    performance_logger = logging.getLogger("performance")
    performance_logger.setLevel(logging.INFO)
    performance_logger.propagate = True
    performance_logger.handlers.clear()


def get_logger(name: str) -> logging.Logger:
    """Return a standard logger."""
    return logging.getLogger(name)
