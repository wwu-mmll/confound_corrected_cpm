import sys
import logging


def setup_logging(log_file: str = "analysis_log.txt"):
    # Get the root logger
    logger = logging.getLogger()

    # Check if handlers already exist and remove them to avoid duplication
    if logger.hasHandlers():
        logger.handlers.clear()

    # Console handler: logs all levels (DEBUG and above) to the console
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(SimpleFormatter())

    # File handler: logs only INFO level logs to the file
    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setLevel(logging.INFO)
    file_handler.addFilter(lambda record: record.levelno == logging.INFO)
    file_handler.setFormatter(SimpleFormatter())

    # Create a logger and set the base level to DEBUG so both handlers can operate independently
    logger.setLevel(logging.DEBUG)  # This ensures all messages are passed to handlers
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)


class SimpleFormatter(logging.Formatter):
    def format(self, record):
        log_fmt = "%(message)s"
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)