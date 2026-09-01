"""
Structured logging setup for the agentic RAG application.
Usage: 
    from utils.logging import get_logger

    logger = get_logger(__name__)
    logger.info("Retirieval started", query=query, top_k=top_k)
"""

import logging
import sys
from pathlib import Path

from loguru import logger as _loguru_logger

class InterceptHandler(logging.Handler):
    """
    Routes stdlib logging records (e.g. from FastAPI, uvicorn, LangChain)
    into loguru so everything end up in one consistent format/sink.
    """
    def emit(self, record: logging.LogRecord) -> None:
        try:
            level = _loguru_logger.level(record.levelname).name
        except:
            level = record.levelno
        frame, depth = logging.currentframe(), 2

        while frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        _loguru_logger.opt(depth=depth, exception=record.exc_info).log(
            level, record.getMessage()
        )

def configure_logging(log_dir: str = "logs", level: str = "INFO") -> None:
    """
    Call once at app startup (e.g. top of app/main.py or rag.py)
    Sets up console + rotating file sinks and captures stdlib logging.
    """
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    _loguru_logger.remove()

    _loguru_logger.add(
        sys.stderr, 
        level=level, 
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> "
        ),
        colorize=True, 
        backtrace=False, 
        diagnose=False
        )

    _loguru_logger.add(
        f"{log_dir}/app.jsonl", 
        level=level, 
        serialize=True, 
        rotation = "10 MB",
        retention = "14 days", 
        compression = "zip",
        backtrace = False, 
        diagnose = False
    )

    logging.basicConfig(
        handlers=[InterceptHandler()], level=0, force=True
    )
    for noisy_logger in ("uvicorn", "uvicorn.error", "uvicorn.access", "htppx"):
        logging.getLogger(noisy_logger).handlers = [InterceptHandler()]

def get_logger(name: str):
    """
    Returns a logger bound with a 'name' field so log lines are traceable
    to a module that emitted them (e.g. "agent.graph", "rag.retriever")
    """
    return _loguru_logger.bind(module=name)