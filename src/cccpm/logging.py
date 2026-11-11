"""Logging setup for CPM analysis runs."""
import sys
import logging


def setup_logging(log_file: str = "analysis_log.txt"):
    """Configures the root logger with a console handler (DEBUG+) and a file handler (INFO only)."""
    logger = logging.getLogger()

    # Clear any handlers from a previous call to avoid duplicate log lines.
    if logger.hasHandlers():
        logger.handlers.clear()

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(SimpleFormatter())

    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setLevel(logging.INFO)
    file_handler.addFilter(lambda record: record.levelno == logging.INFO)
    file_handler.setFormatter(SimpleFormatter())

    # Root level must stay at DEBUG so both handlers can filter independently.
    logger.setLevel(logging.DEBUG)
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)


class SimpleFormatter(logging.Formatter):
    """Formatter that emits only the raw message, with no level/time/name prefix."""

    def format(self, record):
        log_fmt = "%(message)s"
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)
