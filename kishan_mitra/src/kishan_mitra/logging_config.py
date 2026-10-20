
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOG_DIR = PROJECT_ROOT / "logs"
LOG_FILE = LOG_DIR / "kishan_mitra.log"

MAX_LOG_SIZE = 2 * 1024 * 1024  # 2 MB
BACKUP_COUNT = 3


def setup_logging(level=logging.INFO):
    """Configure console and rotating-file logging."""

    LOG_DIR.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("kishan_mitra")
    logger.setLevel(level)
    logger.propagate = False

    # Avoid adding duplicate handlers if setup_logging is called again.
    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)

    file_handler = RotatingFileHandler(
        LOG_FILE,
        maxBytes=MAX_LOG_SIZE,
        backupCount=BACKUP_COUNT,
        encoding="utf-8",
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    logger.info("Logging initialized.")

    return logger
