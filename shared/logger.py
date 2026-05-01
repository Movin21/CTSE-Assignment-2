import logging


def _build_logger() -> logging.Logger:
    logger = logging.getLogger("swarm_trace")
    logger.setLevel(logging.DEBUG)
    if not logger.handlers:
        fmt = logging.Formatter(
            "%(asctime)s | %(levelname)-12s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        file_handler = logging.FileHandler("trace.log", encoding="utf-8")
        file_handler.setFormatter(fmt)
        file_handler.setLevel(logging.DEBUG)
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(fmt)
        console_handler.setLevel(logging.INFO)
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
    return logger


_logger = _build_logger()


def log_event(event_type: str, agent: str, details: str) -> None:
    """Unified logger used by agents and tools alike."""
    _logger.info("[%-16s] agent=%-20s %s", event_type, agent, details)
