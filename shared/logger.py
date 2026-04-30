import logging
from typing import Any


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


def log_tool_call(tool_name: str, inputs: dict[str, Any], output: Any) -> None:
    """Log a tool invocation and its structured result for tracing.

    Args:
        tool_name: Registered name of the tool.
        inputs: Keyword arguments passed to the tool.
        output: Value returned by the tool (may be large).
    """
    if isinstance(output, list):
        preview = f"list[len={len(output)}]"
    else:
        preview = repr(output)
    log_event("TOOL_CALL", tool_name, f"inputs={inputs!r} {preview}")


def log_agent_start(agent_name: str) -> None:
    """Mark the start of an agent node in the trace log.

    Args:
        agent_name: Human-readable agent identifier.
    """
    log_event("AGENT_START", agent_name, "→ entering node")


def log_agent_end(agent_name: str, duration: float, summary: dict[str, Any]) -> None:
    """Mark the end of an agent node with timing and summary metadata.

    Args:
        agent_name: Human-readable agent identifier.
        duration: Elapsed wall time in seconds for the node.
        summary: Small structured dict (e.g. counts) for observability.
    """
    log_event(
        "AGENT_END",
        agent_name,
        f"← exiting node duration={duration:.3f}s summary={summary!r}",
    )
